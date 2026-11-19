// HeteroFlash Bet 1 dual-memory runtime (2026-11-17; 2026-11-18: GPU wait timers + gated pack/wait for deferral): graph-capturable GPU <-> workstation-CPU exchange + GPU Q4_K
// routed experts, so every MoE layer reads both memories at once.
//
// Mailbox v2 (extends ops/route3/remote_hook/hf_mailbox.cu, same HFX1/HFY1 wire protocol):
//   - the reply slot is fp32; with flag bit2 the server returns fp32 partial sums (default), otherwise bf16 (converted
//     to fp32 by the relay); wait_add adds the fp32 partial to the bf16 layer output with one rounding.
//   - the remote mask is a device tensor written by the assign kernel each step (per-step rebalancing), not a plan.
// assign kernel: per MoE layer and step, the union U of routed experts in the window; resident (GPU Q4_K) experts in
//   the union are ranked; the first cap = clamp(round(a*U + b), 0, min(resident, MAXG)) go to the GPU, all others to the
//   workstation CPU. Output: remote[288], the GPU list (pool slots) and per-(slot, token) routing weights.
// Q4_K kernels: llama.cpp block_q4_K (256 values: fp16 d, fp16 dmin, 12 B 6-bit scales/mins, 128 B nibbles), the same
//   bytes the workstation serves; activations fp32 (the CPU quantizes them to q8_K, the only numeric difference);
//   GLM clamped SiLU (gate <= 10, up in [-10, 10]); output = sum_routes w * down(silu(gate x) * up x).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include <stdexcept>

#define HMB_H 4096
#define HMB_I 2048
#define HMB_MAXT 16
#define HMB_K 8
#define NEXP 288
#define MAXG 64
#define KT 4            // tokens per Q4_K kernel pass (fp32 x in shared memory)
#define QK 256
#define BQ4 144
#define GU_ROWS 32      // gate/up rows (warps) per block: one shared x stage feeds 32 warps
#define DN_WARPS 16     // down: warps per block, 2 rows each

// ------------------------------------------------------------------------------------------------ mailbox
struct alignas(64) Slot {
  volatile unsigned long long req_flag;
  volatile unsigned long long done_flag;
  int layer;
  int ntok;
  unsigned short expert[HMB_MAXT * HMB_K];
  float weight[HMB_MAXT * HMB_K];
  alignas(64) __nv_bfloat16 x[HMB_MAXT * HMB_H];   // 16 B aligned: the pack kernel copies it as uint4
  alignas(64) float y[HMB_MAXT * HMB_H];
  unsigned int server_us, relay_us, npairs, pad;
};

// Dense row-split slots (stage M3: lm_head vocabulary rows, later projection rows): request = bf16 x [T, K]; reply =
// the workstation's rows of the output, bf16 [T, N] (the same rounding as a GPU GEMM output).
#define DMAXT 16
#define DMAXSLOT 64
struct alignas(64) DSlot {
  volatile unsigned long long req_flag;
  volatile unsigned long long done_flag;
  int lid, ntok, K, N;
  unsigned int server_us, relay_us;
  alignas(64) __nv_bfloat16 x[DMAXT * HMB_H];
};
static DSlot *h_dslots = nullptr, *d_dslots = nullptr;
static __nv_bfloat16 *h_dy[DMAXSLOT], *d_dy[DMAXSLOT];
static int dN[DMAXSLOT];
static unsigned long long *d_dseq = nullptr;
static int n_dslots = 0;
static std::atomic<long long> st_dreq{0}, st_dus{0}, st_drt_us{0};

static Slot *h_slots = nullptr, *d_slots = nullptr;
static unsigned long long *d_seq = nullptr;
static int n_slots = 0, sock_fd = -1;
static std::thread relay;
static std::atomic<bool> relay_run{false};
static std::atomic<long long> st_req{0}, st_pairs{0}, st_us{0}, st_rt_us{0};
static std::atomic<long long> sd_req{0}, sd_pairs{0}, sd_us{0}, sd_rt_us{0};   // decode-size windows only (T <= 4)

// 2026-11-18 deferral study: GPU-side wait timers (globaltimer ns). Per slot and bucket (0: T <= 4 decode, 1: larger):
// [n, blocked_ns (wait kernel start -> reply seen), overlap_ns (pack -> wait kernel start: GPU work done meanwhile),
//  zero-wait count (reply already there, < 2 us)]. Host-mapped, one writer at a time (stream order), read without CUDA ops.
#define WST 4
static long long *h_wst = nullptr, *d_wst = nullptr;     // [2][n_slots][WST]
static long long *h_dwst = nullptr, *d_dwst = nullptr;   // dense: [DMAXSLOT][2] (n, blocked_ns)
static unsigned long long *d_tpack = nullptr;            // [n_slots] pack time
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void pack_kernel(Slot *slots, unsigned long long *seqs, int slot, int layer_idx, const __nv_bfloat16 *x,
                            const int64_t *ids64, const int *ids32, const float *w, const bool *remote, int T,
                            const int *gate, int want, unsigned long long *tpack) {
  if (gate && *gate != want) return;      // 2026-11-18: run-time switch (late-expert slot), CUDA graphs stay valid
  Slot *s = slots + slot;
  int tid = threadIdx.x;
  for (int i = tid; i < T * HMB_K; i += blockDim.x) {
    long long e = ids64 ? (long long)ids64[i] : (long long)ids32[i];
    bool r = (e >= 0 && e < NEXP) ? remote[e] : false;
    s->expert[i] = r ? (unsigned short)e : (unsigned short)0xFFFF;
    s->weight[i] = w[i];
  }
  const uint4 *xs = (const uint4 *)x; uint4 *xd = (uint4 *)s->x;
  for (int i = tid; i < T * HMB_H / 8; i += blockDim.x) xd[i] = xs[i];
  __syncthreads();
  if (tid == 0) {
    s->layer = layer_idx;
    s->ntok = T;
    unsigned long long q = seqs[slot] + 1;
    seqs[slot] = q;
    __threadfence_system();
    s->req_flag = q;
    __threadfence_system();
    if (tpack) tpack[slot] = gtimer();
  }
}

__global__ void wait_add_kernel(Slot *slots, const unsigned long long *seqs, int slot, __nv_bfloat16 *out, int T,
                                const int *gate, int gwant, long long *wst, const unsigned long long *tpack, int nslots) {
  if (gate && *gate != gwant) return;     // 2026-11-18: deferral switch (this wait runs at layer L or at L+1)
  Slot *s = slots + slot;
  __shared__ unsigned long long want;
  if (threadIdx.x == 0) {
    want = seqs[slot];
    unsigned long long t0 = gtimer();
    while (s->done_flag != want) { __nanosleep(100); }
    __threadfence_system();
    if (wst) {
      unsigned long long t1 = gtimer();
      volatile long long *w = wst + ((size_t)(T <= 4 ? 0 : 1) * nslots + slot) * WST;
      long long b = (long long)(t1 - t0);
      w[0] = w[0] + 1; w[1] = w[1] + b; w[2] = w[2] + (long long)(t0 - tpack[slot]); if (b < 2000) w[3] = w[3] + 1;
    }
  }
  __syncthreads();
  const volatile float *y = s->y;
  for (int i = threadIdx.x; i < T * HMB_H; i += blockDim.x) out[i] = __float2bfloat16(__bfloat162float(out[i]) + y[i]);
}

// 2026-11-17 frontier: per-layer "alt" (lower-bit) CPU expert table, set at run time; bit i = MoE layer index i (0 = layer 3).
// Applied to decode/split windows only: prefill chunks are packed with layer_idx | 0x100 and keep the main table.
static std::atomic<unsigned long long> alt_mask{0};
static std::atomic<long long> st_alt{0};
static int keepalive_us = 0;            // >0: send a 0x00 byte when the link has been idle this long (EEE keep-awake)
static double last_tx = 0;
static inline double now_us();
static void writen(const void *b, size_t n) {
  last_tx = now_us();
  const char *p = (const char *)b;
  while (n) { ssize_t k = send(sock_fd, p, n, MSG_NOSIGNAL); if (k <= 0) throw std::runtime_error("send"); p += k; n -= k; }
}
static void readn(void *b, size_t n) {   // busy-poll receive: no sleep/wakeup on the reply path
  char *p = (char *)b;
  while (n) {
    ssize_t k = recv(sock_fd, p, n, MSG_DONTWAIT);
    if (k > 0) { p += k; n -= k; continue; }
    if (k == 0) throw std::runtime_error("recv closed");
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) throw std::runtime_error("recv");
  }
}
static inline double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static inline float bf2f(unsigned short b) { unsigned int u = (unsigned int)b << 16; float f; memcpy(&f, &u, 4); return f; }

static void dense_one(DSlot *s, int si, int flags) {
  double t0 = now_us();
  int T = s->ntok, K = s->K, N = s->N;
  unsigned int magic = 0x31444648u;
  unsigned short hdr[4] = {(unsigned short)s->lid, (unsigned short)T, (unsigned short)flags, 0};
  std::vector<unsigned char> buf(12 + (size_t)T * K * 2);
  memcpy(buf.data(), &magic, 4); memcpy(buf.data() + 4, hdr, 8); memcpy(buf.data() + 12, (const void *)s->x, (size_t)T * K * 2);
  writen(buf.data(), buf.size());
  unsigned char c = 0;
  do { readn(&c, 1); } while (c == 0);
  unsigned char rest[11]; readn(rest, 11);
  if (!(c == 'H' && rest[0] == 'F' && rest[1] == 'E' && rest[2] == '1')) throw std::runtime_error("bad dense reply magic");
  unsigned short rt, rst; unsigned int us; memcpy(&rt, rest + 3, 2); memcpy(&rst, rest + 5, 2); memcpy(&us, rest + 7, 4);
  if (rst != 0 || rt != T) throw std::runtime_error("dense server status");
  readn((void *)h_dy[si], (size_t)T * N * 2);
  s->server_us = us; s->relay_us = (unsigned int)(now_us() - t0);
  st_dreq++; st_dus += us; st_drt_us += s->relay_us;
}

static void relay_loop(int flags) {
  std::vector<unsigned long long> seen(n_slots, 0), dseen(DMAXSLOT, 0);
  std::vector<unsigned char> buf, yb;
  const bool f32 = (flags & 4) != 0;
  while (relay_run.load(std::memory_order_relaxed)) {
    if (keepalive_us > 0 && now_us() - last_tx > keepalive_us) { const char z = 0; send(sock_fd, &z, 1, MSG_NOSIGNAL | MSG_DONTWAIT); last_tx = now_us(); }
    for (int si = 0; si < n_dslots; si++) {
      DSlot *s = h_dslots + si;
      unsigned long long q = s->req_flag;
      if (q == dseen[si]) continue;
      std::atomic_thread_fence(std::memory_order_acquire);
      dense_one(s, si, flags & 3);
      std::atomic_thread_fence(std::memory_order_release);
      s->done_flag = q;
      dseen[si] = q;
    }
    for (int si = 0; si < n_slots; si++) {
      Slot *s = h_slots + si;
      unsigned long long q = s->req_flag;
      if (q == seen[si]) continue;
      std::atomic_thread_fence(std::memory_order_acquire);
      double t0 = now_us();
      int T = s->ntok;
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
      memset((void *)s->y, 0, sizeof(float) * T * HMB_H);
      unsigned int us = 0;
      if (nt > 0) {
        unsigned int magic = 0x31584648u;
        int lay = s->layer & 0xFF; bool pre = (s->layer & 0x100) != 0;
        bool alt = !pre && ((alt_mask.load(std::memory_order_relaxed) >> lay) & 1ull);
        if (alt) st_alt++;
        unsigned short hdr[4] = {(unsigned short)lay, (unsigned short)nt, (unsigned short)(pairs.size() / 8), (unsigned short)(flags | (alt ? 8 : 0))};
        buf.resize(4 + 8 + pairs.size() + (size_t)nt * HMB_H * 2);
        unsigned char *p = buf.data();
        memcpy(p, &magic, 4); memcpy(p + 4, hdr, 8); memcpy(p + 12, pairs.data(), pairs.size());
        unsigned char *xp = p + 12 + pairs.size();
        for (int t = 0; t < T; t++) if (tok_map[t] >= 0) memcpy(xp + (size_t)tok_map[t] * HMB_H * 2, (const void *)(s->x + (size_t)t * HMB_H), HMB_H * 2);
        writen(buf.data(), buf.size());
        unsigned char c = 0;
        do { readn(&c, 1); } while (c == 0);
        unsigned char rest[11]; readn(rest, 11);
        if (!(c == 'H' && rest[0] == 'F' && rest[1] == 'Y' && rest[2] == '1')) throw std::runtime_error("bad reply magic");
        unsigned short rnt, rst; memcpy(&rnt, rest + 3, 2); memcpy(&rst, rest + 5, 2); memcpy(&us, rest + 7, 4);
        if (rst != 0 || rnt != nt) throw std::runtime_error("expert_server status");
        size_t eb = f32 ? 4 : 2;
        yb.resize((size_t)nt * HMB_H * eb); readn(yb.data(), yb.size());
        for (int t = 0; t < T; t++) {
          if (tok_map[t] < 0) continue;
          float *dst = (float *)(s->y + (size_t)t * HMB_H);
          const unsigned char *src = yb.data() + (size_t)tok_map[t] * HMB_H * eb;
          if (f32) memcpy(dst, src, HMB_H * 4);
          else { const unsigned short *b = (const unsigned short *)src; for (int i = 0; i < HMB_H; i++) dst[i] = bf2f(b[i]); }
        }
        st_req++; st_pairs += pairs.size() / 8; st_us += us;
      }
      s->server_us = us; s->relay_us = (unsigned int)(now_us() - t0); s->npairs = (unsigned int)(pairs.size() / 8);
      st_rt_us += s->relay_us;
      if (T <= 4) { sd_req++; sd_pairs += s->npairs; sd_us += us; sd_rt_us += s->relay_us; }
      std::atomic_thread_fence(std::memory_order_release);
      s->done_flag = q;
      seen[si] = q;
    }
  }
}

void mb_init(int64_t slots, std::string host, int64_t port, int64_t flags) {
  if (h_slots) return;
  if (getenv("HF_SPLIT_KEEPALIVE_US")) keepalive_us = atoi(getenv("HF_SPLIT_KEEPALIVE_US"));
  n_slots = (int)slots;
  if (cudaHostAlloc((void **)&h_slots, sizeof(Slot) * n_slots, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc");
  memset((void *)h_slots, 0, sizeof(Slot) * n_slots);
  if (cudaHostGetDevicePointer((void **)&d_slots, h_slots, 0) != cudaSuccess) throw std::runtime_error("cudaHostGetDevicePointer");
  if (cudaMalloc((void **)&d_seq, sizeof(unsigned long long) * n_slots) != cudaSuccess) throw std::runtime_error("cudaMalloc");
  if (cudaHostAlloc((void **)&h_dslots, sizeof(DSlot) * DMAXSLOT, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc d");
  memset((void *)h_dslots, 0, sizeof(DSlot) * DMAXSLOT);
  if (cudaHostGetDevicePointer((void **)&d_dslots, h_dslots, 0) != cudaSuccess) throw std::runtime_error("dev ptr d");
  if (cudaMalloc((void **)&d_dseq, sizeof(unsigned long long) * DMAXSLOT) != cudaSuccess) throw std::runtime_error("cudaMalloc d");
  cudaMemset(d_dseq, 0, sizeof(unsigned long long) * DMAXSLOT);
  cudaMemset(d_seq, 0, sizeof(unsigned long long) * n_slots);
  if (cudaHostAlloc((void **)&h_wst, sizeof(long long) * 2 * n_slots * WST, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc wst");
  memset(h_wst, 0, sizeof(long long) * 2 * n_slots * WST); cudaHostGetDevicePointer((void **)&d_wst, h_wst, 0);
  if (cudaHostAlloc((void **)&h_dwst, sizeof(long long) * DMAXSLOT * 2, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc dwst");
  memset(h_dwst, 0, sizeof(long long) * DMAXSLOT * 2); cudaHostGetDevicePointer((void **)&d_dwst, h_dwst, 0);
  if (cudaMalloc((void **)&d_tpack, sizeof(unsigned long long) * n_slots) != cudaSuccess) throw std::runtime_error("cudaMalloc tpack");
  cudaMemset(d_tpack, 0, sizeof(unsigned long long) * n_slots);
  cudaDeviceSynchronize();
  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  { int us = 50; int rc = setsockopt(sock_fd, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof us);   // kernel busy-poll on the reply path
    fprintf(stderr, "[hf-drt] relay SO_BUSY_POLL %d us: rc %d (%s)\n", us, rc, rc ? strerror(errno) : "ok"); }
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); inet_pton(AF_INET, host.c_str(), &a.sin_addr);
  if (connect(sock_fd, (sockaddr *)&a, sizeof a)) throw std::runtime_error("connect expert_server");
  relay_run = true;
  relay = std::thread(relay_loop, (int)flags);
  relay.detach();
}

static const int *gate_ptr(torch::Tensor g);
void mb_pack(int64_t slot, int64_t layer_idx, torch::Tensor x, torch::Tensor ids, torch::Tensor w, torch::Tensor remote, torch::Tensor gate, int64_t want) {
  int T = (int)x.size(0);
  TORCH_CHECK(T <= HMB_MAXT && ids.size(1) == HMB_K && x.size(1) == HMB_H && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && ids.is_contiguous());
  bool is64 = ids.scalar_type() == at::kLong;
  pack_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      d_slots, d_seq, (int)slot, (int)layer_idx, (const __nv_bfloat16 *)x.data_ptr(), is64 ? ids.data_ptr<int64_t>() : nullptr,
      is64 ? nullptr : ids.data_ptr<int>(), w.data_ptr<float>(), remote.data_ptr<bool>(), T, gate_ptr(gate), (int)want, d_tpack);
}

void mb_wait_add(int64_t slot, torch::Tensor out, torch::Tensor gate, int64_t want) {
  int T = (int)out.size(0);
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && T <= HMB_MAXT);
  wait_add_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(d_slots, d_seq, (int)slot, (__nv_bfloat16 *)out.data_ptr(), T,
      gate_ptr(gate), (int)want, d_wst, d_tpack, n_slots);
}
// flat [2][n_slots][WST]: n, blocked_ns, overlap_ns, zero-wait count (host memory only: no CUDA call)
std::vector<int64_t> mb_wait_stats() { std::vector<int64_t> v(2 * n_slots * WST); for (size_t i = 0; i < v.size(); i++) v[i] = ((volatile long long *)h_wst)[i]; return v; }
std::vector<int64_t> dense_wait_stats() { std::vector<int64_t> v(2 * n_dslots); for (size_t i = 0; i < v.size(); i++) v[i] = ((volatile long long *)h_dwst)[i]; return v; }

std::vector<int64_t> mb_stats() { return {st_req.load(), st_pairs.load(), st_us.load(), st_rt_us.load(),
                                          sd_req.load(), sd_pairs.load(), sd_us.load(), sd_rt_us.load(), st_alt.load()}; }
void mb_set_alt(int64_t mask) { alt_mask.store((unsigned long long)mask); }
std::vector<int64_t> mb_slot_info(int64_t slot) { Slot *s = h_slots + slot; return {(int64_t)s->server_us, (int64_t)s->relay_us, (int64_t)s->npairs}; }

// ------------------------------------------------------------------------------------------------ expert drop (2026-11-17)
// Training-free routed-expert dropping, one thread per token, graph-capturable (parameters read from device memory):
// expert i of a token is dropped when rank_i >= kmax[layer] or w_i < tau * sum_j w_j (rank by weight, ties by slot).
// Dropped: id -> -1 (assign/pack/relay skip it on both devices), weight -> 0. renorm != 0 rescales the kept weights to
// the original sum ("active"); 0 keeps the original top-8 normalisation ("ref8").
__global__ void drop_kernel(const int64_t *ids64, const int *ids32, const float *w, int T, const int *kmax, int layer,
                            const float *tau_rn, int64_t *o64, int *o32, float *ow) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const int K = kmax[layer]; const float tau = tau_rn[0], rn = tau_rn[1];
  float wv[HMB_K], tot = 0.f, kept = 0.f; bool keep[HMB_K];
#pragma unroll
  for (int i = 0; i < HMB_K; i++) { wv[i] = w[t * HMB_K + i]; tot += wv[i]; }
#pragma unroll
  for (int i = 0; i < HMB_K; i++) {
    int r = 0;
#pragma unroll
    for (int j = 0; j < HMB_K; j++) r += (wv[j] > wv[i]) || (wv[j] == wv[i] && j < i);
    keep[i] = r < K && wv[i] >= tau * tot;
    if (keep[i]) kept += wv[i];
  }
  float sc = (rn != 0.f && kept > 0.f) ? tot / kept : 1.f;
#pragma unroll
  for (int i = 0; i < HMB_K; i++) {
    int k = t * HMB_K + i;
    ow[k] = keep[i] ? wv[i] * sc : 0.f;
    if (o64) o64[k] = keep[i] ? ids64[k] : -1; else o32[k] = keep[i] ? ids32[k] : -1;
  }
}

// 2026-11-18 union-aware drop (decode windows, T <= HMB_MAXT; one block). tau_rn = [tau, renorm, union].
// Core pick = rank < K and w >= tau * sum (the plain rule). union == 0: keep = core (identical to drop_kernel).
// union != 0: a non-core pick is also kept when its expert is in the window union of core picks (any token): the experts
// read stay the core union (cost of plain top-K) and tail picks that cost nothing extra are kept.
__global__ void drop_win_kernel(const int64_t *ids64, const int *ids32, const float *w, int T, const int *kmax, int layer,
                                const float *tau_rn, int64_t *o64, int *o32, float *ow) {
  __shared__ int present[NEXP];
  __shared__ unsigned char core[HMB_MAXT * HMB_K];
  const int K = kmax[layer]; const float tau = tau_rn[0], rn = tau_rn[1]; const bool uni = tau_rn[2] != 0.f;
  int tid = threadIdx.x;
  for (int e = tid; e < NEXP; e += blockDim.x) present[e] = 0;
  __syncthreads();
  if (tid < T) {
    float wv[HMB_K], tot = 0.f;
    for (int i = 0; i < HMB_K; i++) { wv[i] = w[tid * HMB_K + i]; tot += wv[i]; }
    for (int i = 0; i < HMB_K; i++) {
      int r = 0;
      for (int j = 0; j < HMB_K; j++) r += (wv[j] > wv[i]) || (wv[j] == wv[i] && j < i);
      bool c = r < K && wv[i] >= tau * tot;
      core[tid * HMB_K + i] = c;
      long long e = ids64 ? ids64[tid * HMB_K + i] : (long long)ids32[tid * HMB_K + i];
      if (c && e >= 0 && e < NEXP) present[e] = 1;
    }
  }
  __syncthreads();
  if (tid < T) {
    float tot = 0.f, kept = 0.f; bool keep[HMB_K];
    for (int i = 0; i < HMB_K; i++) {
      int k = tid * HMB_K + i; long long e = ids64 ? ids64[k] : (long long)ids32[k];
      tot += w[k];
      keep[i] = core[k] || (uni && e >= 0 && e < NEXP && present[e]);
      if (keep[i]) kept += w[k];
    }
    float sc = (rn != 0.f && kept > 0.f) ? tot / kept : 1.f;
    for (int i = 0; i < HMB_K; i++) {
      int k = tid * HMB_K + i;
      ow[k] = keep[i] ? w[k] * sc : 0.f;
      if (o64) o64[k] = keep[i] ? ids64[k] : -1; else o32[k] = keep[i] ? ids32[k] : -1;
    }
  }
}

void drop(torch::Tensor ids, torch::Tensor w, torch::Tensor kmax, int64_t layer, torch::Tensor tau_rn, torch::Tensor oids, torch::Tensor ow) {
  int T = (int)ids.size(0);
  TORCH_CHECK(ids.size(1) == HMB_K && ids.is_contiguous() && w.is_contiguous() && w.scalar_type() == at::kFloat && oids.scalar_type() == ids.scalar_type());
  bool is64 = ids.scalar_type() == at::kLong;
  if (T == 0) return;
  if (tau_rn.numel() >= 3 && T <= HMB_MAXT) {
    drop_win_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        is64 ? ids.data_ptr<int64_t>() : nullptr, is64 ? nullptr : ids.data_ptr<int>(), w.data_ptr<float>(), T, kmax.data_ptr<int>(), (int)layer,
        tau_rn.data_ptr<float>(), is64 ? oids.data_ptr<int64_t>() : nullptr, is64 ? nullptr : oids.data_ptr<int>(), ow.data_ptr<float>());
    return;
  }
  drop_kernel<<<(T + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      is64 ? ids.data_ptr<int64_t>() : nullptr, is64 ? nullptr : ids.data_ptr<int>(), w.data_ptr<float>(), T, kmax.data_ptr<int>(), (int)layer,
      tau_rn.data_ptr<float>(), is64 ? oids.data_ptr<int64_t>() : nullptr, is64 ? nullptr : oids.data_ptr<int>(), ow.data_ptr<float>());
}

// ------------------------------------------------------------------------------------------------ assign
// stats per layer: [calls, sum U, sum cap, sum resident-in-union]
__global__ void assign_kernel(const int64_t *ids64, const int *ids32, const float *w, int T, const int *slot_of, const float *ab,
                              int mode, bool *remote, int *glist, int *gcount, float *gw, long long *stats) {
  const float a = ab[0], b = ab[1];      // balance rule from device memory: tunable at run time under CUDA graphs
  __shared__ int present[NEXP], rank_res[NEXP];
  __shared__ int s_cap, s_U, s_R;
  int tid = threadIdx.x;
  for (int e = tid; e < NEXP; e += blockDim.x) present[e] = 0;
  __syncthreads();
  for (int i = tid; i < T * HMB_K; i += blockDim.x) {
    long long e = ids64 ? ids64[i] : (long long)ids32[i];
    if (e >= 0 && e < NEXP) present[e] = 1;
  }
  __syncthreads();
  if (tid == 0) {
    int U = 0, R = 0;
    for (int e = 0; e < NEXP; e++) {
      rank_res[e] = -1;
      if (!present[e]) continue;
      U++;
      if (slot_of[e] >= 0) rank_res[e] = R++;
    }
    int cap;
    if (mode == 0) cap = 0;                       // every route to the CPU (config B)
    else if (mode == 2) cap = R;                  // every resident expert on the GPU
    else { float c = a * (float)U + b; cap = c <= 0.f ? 0 : (int)(c + 0.5f); if (cap > R) cap = R; }
    if (cap > MAXG) cap = MAXG;
    s_cap = cap; s_U = U; s_R = R;
  }
  __syncthreads();
  int cap = s_cap;
  for (int e = tid; e < NEXP; e += blockDim.x) {
    bool g = present[e] && rank_res[e] >= 0 && rank_res[e] < cap;
    remote[e] = !g;
    if (g) glist[rank_res[e]] = slot_of[e];
  }
  for (int i = tid; i < MAXG * HMB_MAXT; i += blockDim.x) gw[i] = 0.f;
  __syncthreads();
  for (int i = tid; i < T * HMB_K; i += blockDim.x) {
    long long e = ids64 ? ids64[i] : (long long)ids32[i];
    if (e < 0 || e >= NEXP) continue;
    int r = rank_res[e];
    if (r >= 0 && r < cap) gw[r * HMB_MAXT + i / HMB_K] = w[i];
  }
  if (tid == 0) {
    *gcount = cap;
    if (stats) { atomicAdd((unsigned long long *)&stats[0], 1ull); atomicAdd((unsigned long long *)&stats[1], (unsigned long long)s_U);
                 atomicAdd((unsigned long long *)&stats[2], (unsigned long long)cap); atomicAdd((unsigned long long *)&stats[3], (unsigned long long)s_R); }
  }
}

void assign(torch::Tensor ids, torch::Tensor w, torch::Tensor slot_of, torch::Tensor ab, int64_t mode, torch::Tensor remote,
            torch::Tensor glist, torch::Tensor gcount, torch::Tensor gw, torch::Tensor stats) {
  int T = (int)ids.size(0);
  TORCH_CHECK(T <= HMB_MAXT && ids.size(1) == HMB_K && ids.is_contiguous() && w.is_contiguous() && w.scalar_type() == at::kFloat);
  bool is64 = ids.scalar_type() == at::kLong;
  assign_kernel<<<1, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      is64 ? ids.data_ptr<int64_t>() : nullptr, is64 ? nullptr : ids.data_ptr<int>(), w.data_ptr<float>(), T, slot_of.data_ptr<int>(),
      ab.data_ptr<float>(), (int)mode, remote.data_ptr<bool>(), glist.data_ptr<int>(), gcount.data_ptr<int>(), gw.data_ptr<float>(),
      stats.numel() ? (long long *)stats.data_ptr<int64_t>() : nullptr);
}

// ------------------------------------------------------------------------------------------------ Q4_K kernels
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &sc, int &m) {
  if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
  else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Lane-interleaved Q4_K row dot (bank-conflict-free): for each 256-value block, lane l reads nibble word l of the 128 B
// payload (chunk l/8, bytes 4*(l%8)..+3 -> values chunk*64 + 4*(l%8) + 0..3 (low) and +32 (high)); the 16 B header is
// a broadcast load. Returns the lane's partial sums (caller reduces over the warp).
template <int T, int NB>
__device__ __forceinline__ void q4k_row_lane(const uint8_t *row, const float *xs, int ld, const float *xsum, int lds, int lane,
                                             float *acc) {
  const int ch = lane >> 3, o4 = (lane & 7) * 4;
#pragma unroll 4
  for (int bk = 0; bk < NB; bk++) {
    const uint8_t *base = row + (size_t)bk * BQ4;
    uint4 hdr = *(const uint4 *)base;
    uint32_t q = *(const uint32_t *)(base + 16 + lane * 4);
    const __half *dh = (const __half *)&hdr;
    float d = __half2float(dh[0]), dmin = __half2float(dh[1]);
    int sc0, m0, sc1, m1;
    scale_min_k4(2 * ch, ((const uint8_t *)&hdr) + 4, sc0, m0);
    scale_min_k4(2 * ch + 1, ((const uint8_t *)&hdr) + 4, sc1, m1);
    float l0 = (float)(q & 0xF), l1 = (float)((q >> 8) & 0xF), l2 = (float)((q >> 16) & 0xF), l3 = (float)((q >> 24) & 0xF);
    float h0 = (float)((q >> 4) & 0xF), h1 = (float)((q >> 12) & 0xF), h2 = (float)((q >> 20) & 0xF), h3 = (float)(q >> 28);
    float ds0 = d * (float)sc0, ds1 = d * (float)sc1;
    bool first = (lane & 7) == 0;            // one lane per chunk adds the min term (per sub-block sums of x)
    float dm0 = dmin * (float)m0, dm1 = dmin * (float)m1;
#pragma unroll
    for (int t = 0; t < T; t++) {
      const float *x = xs + (size_t)t * ld + bk * QK + ch * 64 + o4;
      float4 xa = *(const float4 *)x, xb = *(const float4 *)(x + 32);
      float v = ds0 * (l0 * xa.x + l1 * xa.y + l2 * xa.z + l3 * xa.w) + ds1 * (h0 * xb.x + h1 * xb.y + h2 * xb.z + h3 * xb.w);
      if (first) v -= dm0 * xsum[t * lds + bk * 8 + 2 * ch] + dm1 * xsum[t * lds + bk * 8 + 2 * ch + 1];
      acc[t] += v;
    }
  }
}

// gate/up + clamped SiLU. grid (HMB_I / GU_ROWS, MAXG), block 32*GU_ROWS (one warp per row of gate and up).
// x: [T, H] bf16 (T <= KT). h: [MAXG, KT, I] fp32.
template <int T>
__global__ void __launch_bounds__(1024) q4k_gate_up_kernel(const __nv_bfloat16 *x, const int *glist, const int *gcount,
                                                          const unsigned long long *ptrs, float *h) {
  int g = blockIdx.y;
  if (g >= *gcount) return;
  const uint8_t *Gp = (const uint8_t *)ptrs[0], *Up = (const uint8_t *)ptrs[1];
  extern __shared__ float sm[];
  float *xs = sm;                       // [T][H]
  float *xsum = sm + T * HMB_H;         // [T][H/32]
  for (int i = threadIdx.x; i < T * HMB_H; i += blockDim.x) xs[i] = __bfloat162float(x[i]);
  __syncthreads();
  for (int i = threadIdx.x; i < T * (HMB_H / 32); i += blockDim.x) {
    const float *p = xs + (i / (HMB_H / 32)) * HMB_H + (i % (HMB_H / 32)) * 32; float s = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) s += p[k];
    xsum[i] = s;
  }
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * GU_ROWS + warp;
  size_t slot = (size_t)glist[g];
  const size_t rb = (size_t)(HMB_H / QK) * BQ4;
  float ag[T], au[T];
#pragma unroll
  for (int t = 0; t < T; t++) { ag[t] = 0.f; au[t] = 0.f; }
  q4k_row_lane<T, HMB_H / QK>(Gp + (slot * HMB_I + row) * rb, xs, HMB_H, xsum, HMB_H / 32, lane, ag);
  q4k_row_lane<T, HMB_H / QK>(Up + (slot * HMB_I + row) * rb, xs, HMB_H, xsum, HMB_H / 32, lane, au);
#pragma unroll
  for (int t = 0; t < T; t++) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ag[t] += __shfl_xor_sync(0xffffffff, ag[t], o); au[t] += __shfl_xor_sync(0xffffffff, au[t], o); }
  }
  if (lane < T) {
    float gv = 0.f, uv = 0.f;
#pragma unroll
    for (int t = 0; t < T; t++) if (t == lane) { gv = ag[t]; uv = au[t]; }
    gv = fminf(gv, 10.f); uv = fminf(fmaxf(uv, -10.f), 10.f);
    h[((size_t)g * KT + lane) * HMB_I + row] = gv / (1.f + __expf(-gv)) * uv;
  }
}

// down + weighted sum over the GPU experts. grid (HMB_H / (2*DN_WARPS)), block 32*DN_WARPS (2 rows per warp).
template <int T>
__global__ void __launch_bounds__(512) q4k_down_kernel(const float *h, const int *glist, const int *gcount, const float *gw, int t0,
                                                       const unsigned long long *ptrs, float *y) {
  const uint8_t *Dp = (const uint8_t *)ptrs[2];
  extern __shared__ float sm[];
  float *hs = sm;                       // [T][I]
  float *hsum = sm + T * HMB_I;         // [T][I/32]
  int ng = *gcount;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t rb = (size_t)(HMB_I / QK) * BQ4;
  float acc[2][T];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[r][t] = 0.f;
  for (int g = 0; g < ng; g++) {
    __syncthreads();
    for (int i = threadIdx.x; i < T * HMB_I; i += blockDim.x) hs[i] = h[((size_t)g * KT + i / HMB_I) * HMB_I + (i % HMB_I)];
    __syncthreads();
    for (int i = threadIdx.x; i < T * (HMB_I / 32); i += blockDim.x) {
      const float *p = hs + (i / (HMB_I / 32)) * HMB_I + (i % (HMB_I / 32)) * 32; float s = 0.f;
#pragma unroll
      for (int k = 0; k < 32; k++) s += p[k];
      hsum[i] = s;
    }
    __syncthreads();
    size_t slot = (size_t)glist[g];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      int row = blockIdx.x * (2 * DN_WARPS) + warp * 2 + r;
      float part[T];
#pragma unroll
      for (int t = 0; t < T; t++) part[t] = 0.f;
      q4k_row_lane<T, HMB_I / QK>(Dp + (slot * HMB_H + row) * rb, hs, HMB_I, hsum, HMB_I / 32, lane, part);
#pragma unroll
      for (int t = 0; t < T; t++) acc[r][t] += gw[g * HMB_MAXT + t0 + t] * part[t];
    }
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
#pragma unroll
    for (int t = 0; t < T; t++) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r][t] += __shfl_xor_sync(0xffffffff, acc[r][t], o);
    }
    int row = blockIdx.x * (2 * DN_WARPS) + warp * 2 + r;
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < T; t++) y[(size_t)t * HMB_H + row] = acc[r][t];
    }
  }
}

template <int T>
static void launch_q4k(const __nv_bfloat16 *x, const int *glist, const int *gcount, const float *gw, int t0,
                       const unsigned long long *ptrs, float *h, float *y, cudaStream_t st) {
  size_t sm1 = (size_t)T * HMB_H * 4 + (size_t)T * (HMB_H / 32) * 4;
  size_t sm2 = (size_t)T * HMB_I * 4 + (size_t)T * (HMB_I / 32) * 4;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(q4k_gate_up_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm1);
    cudaFuncSetAttribute(q4k_down_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2);
    attr = true;
  }
  q4k_gate_up_kernel<T><<<dim3(HMB_I / GU_ROWS, MAXG), 32 * GU_ROWS, sm1, st>>>(x, glist, gcount, ptrs, h);
  q4k_down_kernel<T><<<HMB_H / (2 * DN_WARPS), 32 * DN_WARPS, sm2, st>>>(h, glist, gcount, gw, t0, ptrs, y);
}

// y[T, H] fp32 = GPU-expert part of the routed sum. x [T, H] bf16 (T <= 16, done in passes of <= KT tokens).
// ptrs: int64 device tensor [3] = base addresses of the G, U, D pools (read by the kernels at run time, so a CUDA graph
// captured before the resident pool exists picks it up once the pointers are written).
void q4k_moe(torch::Tensor x, torch::Tensor glist, torch::Tensor gcount, torch::Tensor gw, torch::Tensor ptrs,
             torch::Tensor hbuf, torch::Tensor y) {
  int T = (int)x.size(0);
  TORCH_CHECK(T <= HMB_MAXT && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && y.scalar_type() == at::kFloat);
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16 *xp = (const __nv_bfloat16 *)x.data_ptr();
  for (int t0 = 0; t0 < T; t0 += KT) {
    int n = std::min(KT, T - t0);
    const __nv_bfloat16 *xc = xp + (size_t)t0 * HMB_H; float *yc = y.data_ptr<float>() + (size_t)t0 * HMB_H;
    const int *gl = glist.data_ptr<int>(), *gc = gcount.data_ptr<int>(); const float *w = gw.data_ptr<float>();
    const unsigned long long *pp = (const unsigned long long *)ptrs.data_ptr<int64_t>(); float *hp = hbuf.data_ptr<float>();
    switch (n) {
      case 1: launch_q4k<1>(xc, gl, gc, w, t0, pp, hp, yc, st); break;
      case 2: launch_q4k<2>(xc, gl, gc, w, t0, pp, hp, yc, st); break;
      case 3: launch_q4k<3>(xc, gl, gc, w, t0, pp, hp, yc, st); break;
      default: launch_q4k<4>(xc, gl, gc, w, t0, pp, hp, yc, st); break;
    }
  }
}

// ------------------------------------------------------------------------------------------------ dense row split
// Register a dense slot (before mb_init's relay sees it: call right after mb_init, outside graph capture).
int64_t dense_register(int64_t lid, int64_t K, int64_t N) {
  int si = n_dslots;
  TORCH_CHECK(si < DMAXSLOT && K <= HMB_H);
  if (cudaHostAlloc((void **)&h_dy[si], (size_t)DMAXT * N * 2, cudaHostAllocMapped) != cudaSuccess) throw std::runtime_error("cudaHostAlloc dy");
  if (cudaHostGetDevicePointer((void **)&d_dy[si], h_dy[si], 0) != cudaSuccess) throw std::runtime_error("dev ptr dy");
  h_dslots[si].lid = (int)lid; h_dslots[si].K = (int)K; h_dslots[si].N = (int)N; dN[si] = (int)N;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  n_dslots = si + 1;
  return si;
}

__global__ void dense_pack_kernel(DSlot *slots, unsigned long long *seqs, int si, const __nv_bfloat16 *x, int T, int K, const int *gate) {
  if (gate && *gate == 0) return;          // run-time switch: this split is off, the GPU computes every row
  DSlot *s = slots + si;
  const uint4 *xs = (const uint4 *)x; uint4 *xd = (uint4 *)s->x;
  for (int i = threadIdx.x; i < T * K / 8; i += blockDim.x) xd[i] = xs[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    s->ntok = T;
    unsigned long long q = seqs[si] + 1; seqs[si] = q;
    __threadfence_system(); s->req_flag = q; __threadfence_system();
  }
}

// out[t, col0 + j] = y[t, j] for j < N (row stride ldo); every block waits for the reply flag.
__global__ void dense_wait_copy_kernel(DSlot *slots, const unsigned long long *seqs, int si, const __nv_bfloat16 *y, int T, int N,
                                       __nv_bfloat16 *out, int ldo, int col0, int ycol0, int nc, const int *gate, long long *dwst) {
  if (gate && *gate == 0) return;
  DSlot *s = slots + si;
  __shared__ unsigned long long want;
  if (threadIdx.x == 0) {
    want = seqs[si]; unsigned long long t0 = gtimer();
    while (s->done_flag != want) { __nanosleep(100); }
    __threadfence_system();
    if (dwst && blockIdx.x == 0) { volatile long long *w = dwst + (size_t)si * 2; w[0] = w[0] + 1; w[1] = w[1] + (long long)(gtimer() - t0); }
  }
  __syncthreads();
  const volatile unsigned short *yv = (const volatile unsigned short *)y;
  unsigned short *o = (unsigned short *)out;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)T * nc; i += (size_t)gridDim.x * blockDim.x) {
    int t = (int)(i / nc), j = (int)(i % nc);
    o[(size_t)t * ldo + col0 + j] = yv[(size_t)t * N + ycol0 + j];
  }
}

static const int *gate_ptr(torch::Tensor g) { return g.numel() ? g.data_ptr<int>() : nullptr; }
void dense_pack(int64_t si, torch::Tensor x, torch::Tensor gate) {
  int T = (int)x.size(0), K = (int)x.size(1);
  TORCH_CHECK(T <= DMAXT && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && K == h_dslots[si].K);
  dense_pack_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(d_dslots, d_dseq, (int)si, (const __nv_bfloat16 *)x.data_ptr(), T, K, gate_ptr(gate));
}

// out[:, col0 : col0 + nc] = reply[:, ycol0 : ycol0 + nc]  (nc < 0: the whole reply)
void dense_wait_copy(int64_t si, int64_t T, torch::Tensor out, int64_t col0, int64_t ycol0, int64_t nc, torch::Tensor gate) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1);
  int N = dN[si];
  if (nc < 0) nc = N - ycol0;
  int blocks = (int)std::min<int64_t>(64, (T * nc + 1023) / 1024);
  dense_wait_copy_kernel<<<std::max(blocks, 1), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(d_dslots, d_dseq, (int)si, d_dy[si], (int)T, N,
      (__nv_bfloat16 *)out.data_ptr(), (int)out.stride(0), (int)col0, (int)ycol0, (int)nc, gate_ptr(gate), d_dwst);
}

// MXFP8 (E4M3 + E8M0 per 32 along K) row GEMV: out[t, r] = bf16(sum_k W[r,k] * 2^(S[r,k/32]-127) * x[t,k]), r < N.
// One warp per row, 32 rows per block; lane reads 4 weights per 128-column step (coalesced), x fp32 in shared memory.
template <int T>
__global__ void __launch_bounds__(1024) mxfp8_gemv_kernel(const uint8_t *W, const uint8_t *S, int N, int K, const __nv_bfloat16 *x,
                                                         __nv_bfloat16 *out, int ldo, const int *gate, int want) {
  if (gate && *gate != want) return;
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < T * K; i += blockDim.x) xs[i] = __bfloat162float(x[i]);
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = blockIdx.x * 32 + warp;
  if (r >= N) return;
  const uint8_t *w = W + (size_t)r * K; const uint8_t *sc = S + (size_t)r * (K / 32);
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = 0.f;
#pragma unroll 4
  for (int k0 = 0; k0 < K; k0 += 128) {
    int k = k0 + lane * 4;
    uint32_t q = *(const uint32_t *)(w + k);
    float s = __int_as_float(((int)sc[k >> 5]) << 23);
    __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q & 0xFFFF), __NV_E4M3);
    __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q >> 16), __NV_E4M3);
    float2 f01 = __half22float2(*(__half2 *)&h01), f23 = __half22float2(*(__half2 *)&h23);
#pragma unroll
    for (int t = 0; t < T; t++) {
      float4 xv = *(const float4 *)(xs + (size_t)t * K + k);
      acc[t] += s * (f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w);
    }
  }
#pragma unroll
  for (int t = 0; t < T; t++) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < T; t++) out[(size_t)t * ldo + r] = __float2bfloat16(acc[t]);
  }
}

template <int T>
static void launch_mx(const uint8_t *W, const uint8_t *S, int N, int K, const __nv_bfloat16 *x, __nv_bfloat16 *out, int ldo, cudaStream_t st,
                      const int *gate, int want) {
  size_t sm = (size_t)T * K * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(mxfp8_gemv_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); attr = true; }
  mxfp8_gemv_kernel<T><<<(N + 31) / 32, 1024, sm, st>>>(W, S, N, K, x, out, ldo, gate, want);
}

// out[:, :N] (row stride out.stride(0)) for x [T, K] bf16, T <= 16 (passes of <= 4 tokens).
// gate (int32 device tensor, may be empty) and want: the kernel runs only when *gate == want (run-time switches).
void mxfp8_gemv(torch::Tensor W, torch::Tensor S, torch::Tensor x, torch::Tensor out, torch::Tensor gate, int64_t want) {
  int N = (int)W.size(0), K = (int)W.size(1), T = (int)x.size(0);
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && K % 128 == 0);
  auto st = at::cuda::getCurrentCUDAStream();
  int ldo = (int)out.stride(0);
  const uint8_t *Wp = (const uint8_t *)W.data_ptr(), *Sp = (const uint8_t *)S.data_ptr();
  for (int t0 = 0; t0 < T; t0 += 4) {
    int n = std::min(4, T - t0);
    const __nv_bfloat16 *xp = (const __nv_bfloat16 *)x.data_ptr() + (size_t)t0 * K;
    __nv_bfloat16 *op = (__nv_bfloat16 *)out.data_ptr() + (size_t)t0 * ldo;
    switch (n) {
      case 1: launch_mx<1>(Wp, Sp, N, K, xp, op, ldo, st, gate_ptr(gate), (int)want); break;
      case 2: launch_mx<2>(Wp, Sp, N, K, xp, op, ldo, st, gate_ptr(gate), (int)want); break;
      case 3: launch_mx<3>(Wp, Sp, N, K, xp, op, ldo, st, gate_ptr(gate), (int)want); break;
      default: launch_mx<4>(Wp, Sp, N, K, xp, op, ldo, st, gate_ptr(gate), (int)want); break;
    }
  }
}

std::vector<int64_t> dense_stats() { return {st_dreq.load(), st_dus.load(), st_drt_us.load()}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mb_init", &mb_init); m.def("mb_pack", &mb_pack); m.def("mb_wait_add", &mb_wait_add); m.def("mb_stats", &mb_stats);
  m.def("mb_slot_info", &mb_slot_info); m.def("mb_set_alt", &mb_set_alt); m.def("drop", &drop); m.def("assign", &assign); m.def("q4k_moe", &q4k_moe);
  m.def("dense_register", &dense_register); m.def("dense_pack", &dense_pack); m.def("dense_wait_copy", &dense_wait_copy);
  m.def("mxfp8_gemv", &mxfp8_gemv); m.def("dense_stats", &dense_stats);
  m.def("mb_wait_stats", &mb_wait_stats); m.def("dense_wait_stats", &dense_wait_stats);
}
