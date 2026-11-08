// HeteroFlash runtime-v2 mailbox: graph-capturable GPU <-> workstation-CPU expert handoff (no Python, no global sync).
//
// Per MoE layer ("slot") a pinned, host-mapped mailbox holds one request and one reply (<= 16 tokens):
//   pack kernel (GPU, after the router): seq = ++dev_seq[slot]; writes T, the 8 (expert, weight) route slots per token
//     (expert = 0xFFFF when that expert is GPU-resident), the bf16 hidden rows; __threadfence_system(); req_flag = seq.
//     It also emits the GPU-side weights with remote slots zeroed (exact b12x mask).
//   relay thread (Spark CPU, C++): spins over slots; on a new req_flag, compacts remote pairs/tokens, sends an HFX1
//     request to expert_server over TCP (busy-poll + heartbeat flags), receives the partial sums, scatters them into
//     the reply rows (zeros for tokens without remote routes), fence, done_flag = seq.
//   wait_add kernel (GPU, queued after the GPU's routed + shared experts): spins until done_flag == dev_seq[slot],
//     then out[t] += reply[t].
// Ordering inside a CUDA graph replay is preserved because dev_seq lives in device memory and only the kernels touch it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include <stdexcept>

#define HMB_H 4096
#define HMB_MAXT 16
#define HMB_K 8

struct alignas(64) Slot {
  volatile unsigned long long req_flag;
  volatile unsigned long long done_flag;
  int layer;          // table index sent to expert_server (abs layer - 3)
  int ntok;
  unsigned short expert[HMB_MAXT * HMB_K];
  float weight[HMB_MAXT * HMB_K];
  __nv_bfloat16 x[HMB_MAXT * HMB_H];
  __nv_bfloat16 y[HMB_MAXT * HMB_H];
};

static Slot *h_slots = nullptr;     // host view
static Slot *d_slots = nullptr;     // device view of the same pinned memory
static unsigned long long *d_seq = nullptr;
static int n_slots = 0;
static std::thread relay;
static std::atomic<bool> relay_run{false};
static std::atomic<long long> st_req{0}, st_pairs{0}, st_us{0};

__global__ void pack_kernel(Slot *slots, unsigned long long *seqs, int slot, int layer_idx, const __nv_bfloat16 *x,
                            const int64_t *ids, int ids_is_64, const int *ids32, const float *w, const bool *remote,
                            float *w_gpu, int T) {
  Slot *s = slots + slot;
  int tid = threadIdx.x;
  for (int i = tid; i < T * HMB_K; i += blockDim.x) {
    long long e = ids_is_64 ? (long long)ids[i] : (long long)ids32[i];
    bool r = (e >= 0 && e < 288) ? remote[e] : false;
    s->expert[i] = r ? (unsigned short)e : (unsigned short)0xFFFF;
    s->weight[i] = w[i];
    w_gpu[i] = r ? 0.0f : w[i];
  }
  for (int i = tid; i < T * HMB_H; i += blockDim.x) s->x[i] = x[i];
  __syncthreads();
  if (tid == 0) {
    s->layer = layer_idx;
    s->ntok = T;
    unsigned long long q = seqs[slot] + 1;
    seqs[slot] = q;
    __threadfence_system();
    s->req_flag = q;
    __threadfence_system();
  }
}

__global__ void wait_add_kernel(Slot *slots, const unsigned long long *seqs, int slot, __nv_bfloat16 *out, int T) {
  Slot *s = slots + slot;
  __shared__ unsigned long long want;
  if (threadIdx.x == 0) {
    want = seqs[slot];
    while (s->done_flag != want) { __nanosleep(200); }
    __threadfence_system();
  }
  __syncthreads();
  const volatile __nv_bfloat16 *y = s->y;
  for (int i = threadIdx.x; i < T * HMB_H; i += blockDim.x) {
    __nv_bfloat16 yv = const_cast<const __nv_bfloat16 &>(y[i]);
    out[i] = __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(yv));
  }
}

static int sock_fd = -1;

static void writen(const void *b, size_t n) {
  const char *p = (const char *)b;
  while (n) { ssize_t k = send(sock_fd, p, n, MSG_NOSIGNAL); if (k <= 0) throw std::runtime_error("send"); p += k; n -= k; }
}
static void readn(void *b, size_t n) {
  char *p = (char *)b;
  while (n) { ssize_t k = recv(sock_fd, p, n, 0); if (k <= 0) throw std::runtime_error("recv"); p += k; n -= k; }
}

static void relay_loop(int flags) {
  std::vector<unsigned long long> seen(n_slots, 0);
  std::vector<unsigned char> buf;
  while (relay_run.load(std::memory_order_relaxed)) {
    bool any = false;
    for (int si = 0; si < n_slots; si++) {
      Slot *s = h_slots + si;
      unsigned long long q = s->req_flag;
      if (q == seen[si]) continue;
      any = true;
      std::atomic_thread_fence(std::memory_order_acquire);
      int T = s->ntok;
      // compact: tokens with at least one remote route, pairs re-indexed to those tokens
      int tok_map[HMB_MAXT], nt = 0;
      std::vector<unsigned char> pairs;
      for (int t = 0; t < T; t++) {
        tok_map[t] = -1;
        for (int k = 0; k < HMB_K; k++) {
          unsigned short e = s->expert[t * HMB_K + k];
          if (e == 0xFFFF) continue;
          if (tok_map[t] < 0) tok_map[t] = nt++;
          unsigned short tt = (unsigned short)tok_map[t];
          float wv = s->weight[t * HMB_K + k];
          unsigned char rec[8]; memcpy(rec, &tt, 2); memcpy(rec + 2, &e, 2); memcpy(rec + 4, &wv, 4);
          pairs.insert(pairs.end(), rec, rec + 8);
        }
      }
      memset((void *)s->y, 0, sizeof(__nv_bfloat16) * T * HMB_H);
      if (nt > 0) {
        unsigned int magic = 0x31584648u;
        unsigned short hdr[4] = {(unsigned short)s->layer, (unsigned short)nt, (unsigned short)(pairs.size() / 8), (unsigned short)flags};
        buf.resize(4 + 8 + pairs.size() + (size_t)nt * HMB_H * 2);
        unsigned char *p = buf.data();
        memcpy(p, &magic, 4); memcpy(p + 4, hdr, 8); memcpy(p + 12, pairs.data(), pairs.size());
        unsigned char *xp = p + 12 + pairs.size();
        for (int t = 0; t < T; t++) if (tok_map[t] >= 0) memcpy(xp + (size_t)tok_map[t] * HMB_H * 2, (const void *)(s->x + (size_t)t * HMB_H), HMB_H * 2);
        writen(buf.data(), buf.size());
        unsigned char c = 0;
        do { readn(&c, 1); } while (c == 0);                 // heartbeat bytes before the reply magic
        unsigned char rest[11]; readn(rest, 11);
        if (!(c == 'H' && rest[0] == 'F' && rest[1] == 'Y' && rest[2] == '1')) throw std::runtime_error("bad reply magic");
        unsigned short rnt, rst; unsigned int us; memcpy(&rnt, rest + 3, 2); memcpy(&rst, rest + 5, 2); memcpy(&us, rest + 7, 4);
        if (rst != 0 || rnt != nt) throw std::runtime_error("expert_server status");
        std::vector<unsigned char> yb((size_t)nt * HMB_H * 2); readn(yb.data(), yb.size());
        for (int t = 0; t < T; t++) if (tok_map[t] >= 0) memcpy((void *)(s->y + (size_t)t * HMB_H), yb.data() + (size_t)tok_map[t] * HMB_H * 2, HMB_H * 2);
        st_req++; st_pairs += pairs.size() / 8; st_us += us;
      }
      std::atomic_thread_fence(std::memory_order_release);
      s->done_flag = q;
      seen[si] = q;
    }
    if (!any) { /* busy spin: decode latency matters more than one Spark core */ }
  }
}

void init(int64_t slots, std::string host, int64_t port, int64_t flags) {
  if (h_slots) return;
  n_slots = (int)slots;
  if (cudaHostAlloc((void **)&h_slots, sizeof(Slot) * n_slots, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc");
  memset((void *)h_slots, 0, sizeof(Slot) * n_slots);
  if (cudaHostGetDevicePointer((void **)&d_slots, h_slots, 0) != cudaSuccess) throw std::runtime_error("cudaHostGetDevicePointer");
  if (cudaMalloc((void **)&d_seq, sizeof(unsigned long long) * n_slots) != cudaSuccess) throw std::runtime_error("cudaMalloc");
  cudaMemset(d_seq, 0, sizeof(unsigned long long) * n_slots);
  cudaDeviceSynchronize();
  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); inet_pton(AF_INET, host.c_str(), &a.sin_addr);
  if (connect(sock_fd, (sockaddr *)&a, sizeof a)) throw std::runtime_error("connect expert_server");
  relay_run = true;
  relay = std::thread(relay_loop, (int)flags);
  relay.detach();
}

torch::Tensor pack(int64_t slot, int64_t layer_idx, torch::Tensor x, torch::Tensor ids, torch::Tensor w, torch::Tensor remote) {
  int T = (int)x.size(0);
  TORCH_CHECK(T <= HMB_MAXT && ids.size(1) == HMB_K && x.size(1) == HMB_H && x.scalar_type() == at::kBFloat16);
  auto wf = w.to(at::kFloat).contiguous();
  auto w_gpu = torch::empty_like(wf);
  auto ic = ids.contiguous();
  bool is64 = ic.scalar_type() == at::kLong;
  pack_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      d_slots, d_seq, (int)slot, (int)layer_idx, (const __nv_bfloat16 *)x.contiguous().data_ptr(),
      is64 ? ic.data_ptr<int64_t>() : nullptr, is64 ? 1 : 0, is64 ? nullptr : ic.data_ptr<int>(),
      wf.data_ptr<float>(), remote.data_ptr<bool>(), w_gpu.data_ptr<float>(), T);
  return w_gpu.to(w.scalar_type());
}

void wait_add(int64_t slot, torch::Tensor out) {
  int T = (int)out.size(0);
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && T <= HMB_MAXT);
  wait_add_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(d_slots, d_seq, (int)slot, (__nv_bfloat16 *)out.data_ptr(), T);
}

std::vector<int64_t> stats() { return {st_req.load(), st_pairs.load(), st_us.load()}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init); m.def("pack", &pack); m.def("wait_add", &wait_add); m.def("stats", &stats);
}
