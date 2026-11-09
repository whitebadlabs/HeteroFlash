// HeteroFlash CPU expert server, GGUF k-quant edition (runtime-v2).
// Same wire protocol as expert_server.c (HFX1 request -> HFY1 reply, flag bit0 busy-poll, bit1 heartbeat), but the
// experts are the rows of a llama.cpp GGUF (e.g. Q4_K + imatrix), memory-mapped, and every dot product is llama.cpp's
// own CPU kernel (ggml_get_type_traits_cpu(type)->vec_dot with activations quantized to its vec_dot_type), so the CPU
// computes exactly what llama.cpp would for that format.
//
// Usage: expert_server_gguf serve <port> <model.gguf> <index.idx>       (index from runtime/gguf_expert_index.py)
//        env HF_BIND=<ip>  OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES as for expert_server
//        expert_server_gguf selftest <model.gguf> <index.idx> <layer> <expert>   (prints a few outputs for comparison)
// Build: gcc -O3 -march=native -fopenmp expert_server_gguf.c -I<llama.cpp>/ggml/include -L<build>/bin -lggml-cpu -lggml-base -lm -lpthread
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <omp.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define H 4096
#define I 2048
#define MAXTOK 16
#define MAXPAIR (MAXTOK * 8)
#define FIRST_MOE 3
#define NLAYERS 43   // 42 MoE layers + MTP layer 45 (table index 42)

typedef struct { int type, rows, cols, ne; size_t row_bytes; const uint8_t *base; } XT;   // one [ne, rows, row_bytes] tensor
static XT tab[NLAYERS][3];
static const struct ggml_type_traits_cpu *tr[GGML_TYPE_COUNT];

static int table_index(int abs_layer) { return abs_layer == 45 ? 42 : abs_layer - FIRST_MOE; }

static int load(const char *gguf, const char *idx) {
    int fd = open(gguf, O_RDONLY); if (fd < 0) { perror("open gguf"); return -1; }
    struct stat st; fstat(fd, &st);
    const uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return -1; }
    madvise((void *)map, st.st_size, MADV_RANDOM);
    FILE *f = fopen(idx, "rb"); if (!f) { perror("open idx"); return -1; }
    int32_t h6[6]; uint64_t q3[3]; int n = 0; size_t bytes = 0;
    while (fread(h6, 4, 6, f) == 6 && fread(q3, 8, 3, f) == 3) {
        int L = table_index(h6[0]); if (L < 0 || L >= NLAYERS) continue;
        XT *t = &tab[L][h6[1]];
        t->type = h6[2]; t->rows = h6[3]; t->cols = h6[4]; t->ne = h6[5]; t->row_bytes = q3[0]; t->base = map + q3[1];
        if (!tr[t->type]) tr[t->type] = ggml_get_type_traits_cpu((enum ggml_type)t->type);
        if (!tr[t->type] || !tr[t->type]->vec_dot) { fprintf(stderr, "no CPU vec_dot for type %d\n", t->type); return -1; }
        n++; bytes += q3[2];
    }
    fclose(f);
    fprintf(stderr, "indexed %d expert tensors, %.1f GB mapped\n", n, bytes / 1e9);
    // prefault: touch every page once (parallel) so the first requests do not stall on disk
    if (getenv("HF_PREFAULT") == NULL || atoi(getenv("HF_PREFAULT"))) {
        double t0 = omp_get_wtime(); volatile uint64_t sink = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+ : sink)
        for (int k = 0; k < NLAYERS * 3; k++) {
            XT *t = &tab[k / 3][k % 3]; if (!t->base) continue;
            size_t nb = (size_t)t->ne * t->rows * t->row_bytes; uint64_t s = 0;
            for (size_t o = 0; o < nb; o += 4096) s += t->base[o];
            sink += s;
        }
        fprintf(stderr, "prefaulted in %.0f s\n", omp_get_wtime() - t0); (void)sink;
    }
    return n;
}

static inline const uint8_t *row_ptr(const XT *t, int e, int r) { return t->base + ((size_t)e * t->rows + r) * t->row_bytes; }

typedef struct { uint16_t tok, expert; float w; } Pair;

// y[ntok][H] = sum over pairs of w * down( silu(gate x) * up x ), all dot products by llama.cpp's vec_dot for the type.
static int run_layer(int L, int ntok, const float *x, int np, const Pair *pairs, float *y) {
    XT *G = &tab[L][0], *U = &tab[L][1], *D = &tab[L][2];
    if (!G->base || !U->base || !D->base) return 1;
    const struct ggml_type_traits_cpu *tg = tr[G->type], *tu = tr[U->type], *td = tr[D->type];
    // quantize activations to each projection's vec_dot_type
    static uint8_t xg[MAXTOK][H * 2], xu[MAXTOK][H * 2];
    size_t rg = ggml_row_size(tg->vec_dot_type, H), ru = ggml_row_size(tu->vec_dot_type, H), rd = ggml_row_size(td->vec_dot_type, I);
    if (rg > sizeof xg[0] || ru > sizeof xu[0]) return 2;
    const struct ggml_type_traits_cpu *qg = ggml_get_type_traits_cpu(tg->vec_dot_type), *qu = ggml_get_type_traits_cpu(tu->vec_dot_type),
                                      *qd = ggml_get_type_traits_cpu(td->vec_dot_type);
    for (int t = 0; t < ntok; t++) {
        qg->from_float(x + (size_t)t * H, xg[t], H);
        if (tu->vec_dot_type == tg->vec_dot_type) memcpy(xu[t], xg[t], rg); else qu->from_float(x + (size_t)t * H, xu[t], H);
    }
    int ne = 0, eid[MAXPAIR], etok[MAXPAIR][MAXTOK], ent[MAXPAIR]; float ew[MAXPAIR][MAXTOK];
    for (int p = 0; p < np; p++) {
        int e = -1;
        for (int k = 0; k < ne; k++) if (eid[k] == pairs[p].expert) { e = k; break; }
        if (e < 0) { e = ne++; eid[e] = pairs[p].expert; ent[e] = 0; }
        if (pairs[p].expert >= G->ne || pairs[p].tok >= ntok) return 1;
        etok[e][ent[e]] = pairs[p].tok; ew[e][ent[e]] = pairs[p].w; ent[e]++;
    }
    static float g[MAXPAIR][MAXTOK][I], u[MAXPAIR][MAXTOK][I];
    static uint8_t hq[MAXPAIR][MAXTOK][I * 2];
    memset(y, 0, sizeof(float) * ntok * H);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) collapse(2)
        for (int e = 0; e < ne; e++) for (int r = 0; r < I; r++) {
            const uint8_t *wg = row_ptr(G, eid[e], r), *wu = row_ptr(U, eid[e], r);
            for (int k = 0; k < ent[e]; k++) {
                tg->vec_dot(H, &g[e][k][r], 0, wg, 0, xg[etok[e][k]], 0, 1);
                tu->vec_dot(H, &u[e][k][r], 0, wu, 0, xu[etok[e][k]], 0, 1);
            }
        }
        #pragma omp for schedule(static) collapse(2)
        for (int e = 0; e < ne; e++) for (int k = 0; k < MAXTOK; k++) {
            if (k >= ent[e]) continue;
            float h[I];
            for (int i = 0; i < I; i++) {   // GLM-5.3 clamped SiLU (swiglu_limit 10): gate <= 10, up in [-10, 10]
                float gv = fminf(g[e][k][i], 10.0f), uv = fminf(fmaxf(u[e][k][i], -10.0f), 10.0f);
                h[i] = gv / (1.0f + expf(-gv)) * uv;
            }
            qd->from_float(h, hq[e][k], I);
        }
        #pragma omp for schedule(static)
        for (int r = 0; r < H; r++) {
            for (int e = 0; e < ne; e++) {
                const uint8_t *wd = row_ptr(D, eid[e], r);
                for (int k = 0; k < ent[e]; k++) {
                    float s; td->vec_dot(I, &s, 0, wd, 0, hq[e][k], 0, 1);
                    y[(size_t)etok[e][k] * H + r] += ew[e][k] * s;
                }
            }
        }
    }
    (void)rd;
    return 0;
}

// ---------------------------------------------------------------- transport (as expert_server.c)
static inline float bf16_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static int busy_conn = 0;
static void set_busy(int fd) {
    int one = 1, us = 50;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof us);
    busy_conn = 1;
}
static int readn(int fd, void *buf, size_t n) {
    size_t o = 0; int spins = 0;
    while (o < n) {
        ssize_t k = recv(fd, (char *)buf + o, n - o, busy_conn ? MSG_DONTWAIT : 0);
        if (k > 0) { o += k; spins = 0; if (busy_conn) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one); } continue; }
        if (k == 0) return -1;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (++spins > 200000) { struct pollfd pf = {.fd = fd, .events = POLLIN}; poll(&pf, 1, 1000); spins = 0; }
            continue;
        }
        if (errno == EINTR) continue;
        return -1;
    }
    return 0;
}
static int writen(int fd, const void *buf, size_t n) {
    size_t o = 0;
    while (o < n) { ssize_t k = send(fd, (const char *)buf + o, n - o, MSG_NOSIGNAL); if (k < 0 && (errno == EAGAIN || errno == EINTR)) continue; if (k <= 0) return -1; o += k; }
    return 0;
}
static atomic_int hb_state = 0, hb_in_send = 0; static int hb_fd = -1;
static void *hb_thread(void *arg) {
    (void)arg; const char z = 0; struct timespec t; double next = 0;
    for (;;) {
        int s = atomic_load(&hb_state); if (s == 2) return NULL;
        clock_gettime(CLOCK_MONOTONIC, &t); double now = t.tv_sec * 1e6 + t.tv_nsec / 1e3;
        if (s == 1 && now >= next) {
            atomic_store(&hb_in_send, 1);
            if (atomic_load(&hb_state) == 1) send(hb_fd, &z, 1, MSG_NOSIGNAL);
            atomic_store(&hb_in_send, 0);
            next = now + 20;
        }
    }
}
static void hb_stop(void) { atomic_store(&hb_state, 0); while (atomic_load(&hb_in_send)) ; }
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

int main(int argc, char **argv) {
    ggml_cpu_init();
    if (argc >= 6 && !strcmp(argv[1], "selftest")) {
        if (load(argv[2], argv[3]) <= 0) return 1;
        int L = table_index(atoi(argv[4])), e = atoi(argv[5]);
        static float x[H], y[H]; for (int i = 0; i < H; i++) x[i] = sinf(i * 0.37f) * (1.0f + (i % 7) * 0.1f) * 0.5f;
        Pair p = {0, (uint16_t)e, 1.0f};
        double t0 = now_us(); int st = run_layer(L, 1, x, 1, &p, y);
        printf("status %d us %.0f y[0..4] %.6f %.6f %.6f %.6f %.6f\n", st, now_us() - t0, y[0], y[1], y[2], y[3], y[4]);
        double ss = 0; for (int i = 0; i < H; i++) ss += (double)y[i] * y[i]; printf("norm %.6f\n", sqrt(ss));
        return 0;
    }
    if (argc < 5 || strcmp(argv[1], "serve")) { fprintf(stderr, "usage: %s serve <port> <gguf> <idx> | selftest <gguf> <idx> <layer> <expert>\n", argv[0]); return 2; }
    int port = atoi(argv[2]);
    if (load(argv[3], argv[4]) <= 0) return 1;
    int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY};
    if (getenv("HF_BIND")) inet_pton(AF_INET, getenv("HF_BIND"), &a.sin_addr);
    if (bind(srv, (void *)&a, sizeof a) || listen(srv, 4)) { perror("bind/listen"); return 1; }
    fprintf(stderr, "listening on %d\n", port);
    pthread_t hbt; int hb_started = 0;
    static float x[MAXTOK * H], y[MAXTOK * H]; static uint16_t xb[MAXTOK * H], yb[MAXTOK * H]; static Pair pairs[MAXPAIR];
    for (;;) {
        int c = accept(srv, NULL, NULL); if (c < 0) continue;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        busy_conn = 0;
        for (;;) {
            uint32_t magic; uint16_t hdr[4];
            if (readn(c, &magic, 4) || magic != 0x31584648u || readn(c, hdr, 8)) break;
            int layer = hdr[0], ntok = hdr[1], np = hdr[2];
            if ((hdr[3] & 1) && !busy_conn) set_busy(c);
            int use_hb = (hdr[3] & 2) != 0;
            if (use_hb && !hb_started) { hb_fd = c; atomic_store(&hb_state, 0); pthread_create(&hbt, NULL, hb_thread, NULL); hb_started = 1; }
            if (ntok > MAXTOK || np > MAXPAIR || layer >= NLAYERS) break;
            struct { uint16_t tok, expert; float w; } __attribute__((packed)) wire[MAXPAIR];
            if (readn(c, wire, (size_t)np * 8) || readn(c, xb, (size_t)ntok * H * 2)) break;
            double t0 = now_us();
            if (use_hb) atomic_store(&hb_state, 1);
            for (int p = 0; p < np; p++) { pairs[p].tok = wire[p].tok; pairs[p].expert = wire[p].expert; pairs[p].w = wire[p].w; }
            for (int i = 0; i < ntok * H; i++) x[i] = bf16_to_f(xb[i]);
            int st = run_layer(layer, ntok, x, np, pairs, y);
            for (int i = 0; i < ntok * H; i++) yb[i] = f_to_bf16(y[i]);
            if (use_hb) hb_stop();
            uint32_t us = (uint32_t)(now_us() - t0), rm = 0x31594648u; uint16_t rh[2] = {(uint16_t)ntok, (uint16_t)st};
            if (writen(c, &rm, 4) || writen(c, rh, 4) || writen(c, &us, 4) || writen(c, yb, (size_t)ntok * H * 2)) break;
        }
        if (hb_started) { atomic_store(&hb_state, 2); pthread_join(hbt, NULL); hb_started = 0; atomic_store(&hb_state, 0); }
        close(c);
    }
}
