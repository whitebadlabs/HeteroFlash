// HeteroFlash remote CPU expert server (workstation side).
// Activation-only protocol over TCP: the Spark sends, per MoE layer, the routed
// (token, expert, gate_weight) pairs whose experts live in this server's RAM plus
// the bf16 hidden states of those tokens; the server returns, per token, the bf16
// gate-weighted sum of those experts' FFN outputs (partial sum to add on the GPU).
//
// Expert FFN: y = down( silu(gate(x)) * up(x) ), int4 weights with an fp32 scale per
// 128-weight group (+ folded zero-point constant), dynamic per-token u8 activations,
// AVX-512 VNNI (see bench/moe_cpu_expert_v3.c). Multi-token batching: each expert's
// weights are streamed once for all tokens routed to it in the request (speculative
// verify windows), which is what makes windows cheaper than separate calls.
//
// Wire format (little-endian):
//   request : u32 magic 'HFX1' | u16 layer | u16 ntok | u16 npairs | u16 reserved
//             npairs x {u16 tok, u16 expert, f32 weight} | ntok x 4096 bf16
//   reply   : u32 magic 'HFY1' | u16 ntok | u16 status (0 ok, 1 missing expert) | u32 server_us | ntok x 4096 bf16
//
// Usage: expert_server serve <port> [synthetic_layers=42] [experts_per_layer=288] [resident_per_layer=288]
//        expert_server selftest
// Weights: synthetic (deterministic random) until the real-weight loader lands.
#include <immintrin.h>
#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define H 4096
#define I 2048
#define G 128
#define MAXTOK 16
#define MAXPAIR (MAXTOK * 8)

typedef struct { uint8_t *q; float *s; float *c; int rows, cols; } QMat;
typedef struct { QMat gu, dn; } Expert;
static Expert **table;          // table[layer][expert] or NULL if not resident
static int NL, NE;

static void qmat_init(QMat *m, int rows, int cols, unsigned seed, int8_t *keep) {
    m->rows = rows; m->cols = cols;
    m->q = aligned_alloc(64, (size_t)rows * cols / 2);
    m->s = aligned_alloc(64, (size_t)rows * (cols / G) * 4);
    m->c = aligned_alloc(64, (size_t)rows * (cols / G) * 4);
    for (int r = 0; r < rows; r++) for (int g = 0; g < cols / G; g++) {
        int sw = 0; float sc = 0.002f + 0.001f * ((r * 7 + g * 13 + seed) % 11);
        for (int j = 0; j < 64; j++) {
            seed = seed * 1103515245u + 12345u; int lo = (seed >> 16) % 16 - 8;
            seed = seed * 1103515245u + 12345u; int hi = (seed >> 16) % 16 - 8;
            m->q[((size_t)r * cols + g * G) / 2 + j] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
            sw += lo + hi;
            if (keep) { keep[(size_t)r * cols + g * G + j] = lo; keep[(size_t)r * cols + g * G + 64 + j] = hi; }
        }
        m->s[r * (cols / G) + g] = sc;
        m->c[r * (cols / G) + g] = sc * (128.0f * sw + 131072.0f);
    }
}

static inline float bf16_to_f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static float quant_act(const float *x, int n, uint8_t *xu, int32_t *sxq) {
    float mx = 1e-12f; for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(x[i]));
    float s = mx / 127.0f, inv = 1.0f / s;
    for (int g = 0; g < n / G; g++) {
        int acc = 0;
        for (int i = g * G; i < (g + 1) * G; i++) { int q = (int)lrintf(x[i] * inv); acc += q; xu[i] = (uint8_t)(q + 128); }
        sxq[g] = acc;
    }
    return s;
}

// 4 output rows x nt tokens: weights streamed once, dotted with every token
static inline void rows4_multi(const QMat *m, int r0, int nt, uint8_t *const *xu, int32_t *const *sxq,
                               const float *sx, float *out /* [nt][4] */) {
    const int ng = m->cols / G; const __m512i mk = _mm512_set1_epi8(0x0f);
    __m512 acc[MAXTOK][4]; float corr[MAXTOK][4];
    for (int t = 0; t < nt; t++) for (int r = 0; r < 4; r++) { acc[t][r] = _mm512_setzero_ps(); corr[t][r] = 0; }
    for (int g = 0; g < ng; g++) {
        __m512i lo[4], hi[4]; float sw[4], cc[4];
        for (int r = 0; r < 4; r++) {
            __m512i p = _mm512_loadu_si512(m->q + ((size_t)(r0 + r) * m->cols + g * G) / 2);
            lo[r] = _mm512_and_si512(p, mk); hi[r] = _mm512_and_si512(_mm512_srli_epi16(p, 4), mk);
            sw[r] = m->s[(r0 + r) * ng + g]; cc[r] = m->c[(r0 + r) * ng + g];
        }
        for (int t = 0; t < nt; t++) {
            __m512i xl = _mm512_loadu_si512(xu[t] + g * G), xh = _mm512_loadu_si512(xu[t] + g * G + 64);
            for (int r = 0; r < 4; r++) {
                __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), xl, lo[r]);
                d = _mm512_dpbusd_epi32(d, xh, hi[r]);
                acc[t][r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_set1_ps(sw[r]), acc[t][r]);
                corr[t][r] += sw[r] * 8.0f * sxq[t][g] + cc[r];
            }
        }
    }
    for (int t = 0; t < nt; t++) for (int r = 0; r < 4; r++) out[t * 4 + r] = sx[t] * (_mm512_reduce_add_ps(acc[t][r]) - corr[t][r]);
}

typedef struct { uint16_t tok, expert; float w; } Pair;

// Computes y[ntok][H] (fp32) = sum over pairs of w * Expert(x[tok]).  Returns 0 or 1 (missing expert).
static int run_layer(int layer, int ntok, const float *x, int npairs, const Pair *pairs, float *y) {
    static uint8_t xu[MAXTOK][H]; static int32_t sxq[MAXTOK][H / G]; static float sx[MAXTOK];
    // group pairs by expert
    int ne = 0, eid[MAXPAIR], etok[MAXPAIR][MAXTOK], ent[MAXPAIR]; float ew[MAXPAIR][MAXTOK];
    for (int p = 0; p < npairs; p++) {
        int e = -1;
        for (int k = 0; k < ne; k++) if (eid[k] == pairs[p].expert) { e = k; break; }
        if (e < 0) { e = ne++; eid[e] = pairs[p].expert; ent[e] = 0; }
        etok[e][ent[e]] = pairs[p].tok; ew[e][ent[e]] = pairs[p].w; ent[e]++;
        if (!table[layer][pairs[p].expert].gu.q) return 1;
    }
    static float gu[MAXPAIR][MAXTOK][2 * I]; static uint8_t hu[MAXPAIR][MAXTOK][I]; static int32_t shq[MAXPAIR][MAXTOK][I / G];
    static float sh[MAXPAIR][MAXTOK];
    memset(y, 0, sizeof(float) * ntok * H);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int t = 0; t < ntok; t++) sx[t] = quant_act(x + (size_t)t * H, H, xu[t], sxq[t]);
        #pragma omp for schedule(static) collapse(2)
        for (int e = 0; e < ne; e++) for (int r = 0; r < 2 * I; r += 4) {
            uint8_t *xp[MAXTOK]; int32_t *sp[MAXTOK]; float sv[MAXTOK], o[MAXTOK * 4];
            for (int k = 0; k < ent[e]; k++) { xp[k] = xu[etok[e][k]]; sp[k] = sxq[etok[e][k]]; sv[k] = sx[etok[e][k]]; }
            rows4_multi(&table[layer][eid[e]].gu, r, ent[e], xp, sp, sv, o);
            for (int k = 0; k < ent[e]; k++) for (int j = 0; j < 4; j++) gu[e][k][r + j] = o[k * 4 + j];
        }
        #pragma omp for schedule(static) collapse(2)
        for (int e = 0; e < ne; e++) for (int k = 0; k < MAXTOK; k++) {
            if (k >= ent[e]) continue;
            float h[I]; const float *g = gu[e][k];
            for (int i = 0; i < I; i++) h[i] = g[i] / (1.0f + expf(-g[i])) * g[I + i];
            sh[e][k] = quant_act(h, I, hu[e][k], shq[e][k]);
        }
        #pragma omp for schedule(static)
        for (int r = 0; r < H; r += 4) {
            for (int e = 0; e < ne; e++) {
                uint8_t *xp[MAXTOK]; int32_t *sp[MAXTOK]; float o[MAXTOK * 4];
                for (int k = 0; k < ent[e]; k++) { xp[k] = hu[e][k]; sp[k] = shq[e][k]; }
                rows4_multi(&table[layer][eid[e]].dn, r, ent[e], xp, sp, sh[e], o);
                for (int k = 0; k < ent[e]; k++) for (int j = 0; j < 4; j++) y[(size_t)etok[e][k] * H + r + j] += ew[e][k] * o[k * 4 + j];
            }
        }
    }
    return 0;
}

static int readn(int fd, void *buf, size_t n) { size_t o = 0; while (o < n) { ssize_t k = read(fd, (char *)buf + o, n - o); if (k <= 0) return -1; o += k; } return 0; }
static int writen(int fd, const void *buf, size_t n) { size_t o = 0; while (o < n) { ssize_t k = write(fd, (const char *)buf + o, n - o); if (k <= 0) return -1; o += k; } return 0; }
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

static void build_table(int layers, int experts, int resident) {
    NL = layers; NE = experts;
    table = calloc(layers, sizeof(Expert *));
    for (int l = 0; l < layers; l++) {
        table[l] = calloc(experts, sizeof(Expert));
        for (int e = 0; e < resident && e < experts; e++) {
            qmat_init(&table[l][e].gu, 2 * I, H, 11 + l * 1000 + e, NULL);
            qmat_init(&table[l][e].dn, H, I, 977 + l * 1000 + e, NULL);
        }
    }
}

// Load exported layer files (src/quantize_experts_int4.py export): <dir>/layer<L>.hfx, L = absolute
// decoder layer (first MoE layer = FIRST_MOE -> table index 0).
#define FIRST_MOE 3
static int load_weights(const char *dir, int layers, int experts) {
    NL = layers; NE = experts;
    table = calloc(layers, sizeof(Expert *));
    for (int l = 0; l < layers; l++) table[l] = calloc(experts, sizeof(Expert));
    int files = 0; size_t bytes = 0;
    for (int l = 0; l < layers; l++) {
        char path[1024]; snprintf(path, sizeof path, "%s/layer%d.hfx", dir, l + FIRST_MOE);
        FILE *f = fopen(path, "rb"); if (!f) continue;
        char magic[4]; uint32_t hl, n;
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "HFQ1", 4) || fread(&hl, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
        uint32_t *ids = malloc(4 * n); if (fread(ids, 4, n, f) != n) { fclose(f); return -1; }
        for (uint32_t k = 0; k < n; k++) {
            Expert *e = &table[l][ids[k]];
            QMat *ms[2] = {&e->gu, &e->dn}; int rows[2] = {2 * I, H}, cols[2] = {H, I};
            for (int m = 0; m < 2; m++) {
                QMat *q = ms[m]; q->rows = rows[m]; q->cols = cols[m];
                size_t nq = (size_t)rows[m] * cols[m] / 2, ns = (size_t)rows[m] * (cols[m] / G);
                q->q = aligned_alloc(64, nq); q->s = aligned_alloc(64, ns * 4); q->c = aligned_alloc(64, ns * 4);
                if (fread(q->q, 1, nq, f) != nq || fread(q->s, 4, ns, f) != ns || fread(q->c, 4, ns, f) != ns) { fclose(f); return -1; }
                bytes += nq + ns * 8;
            }
        }
        free(ids); fclose(f); files++;
        fprintf(stderr, "loaded layer %u: %u experts\n", hl, n);
    }
    fprintf(stderr, "loaded %d layer files, %.1f GB\n", files, bytes / 1e9);
    return files;
}

static int selftest(void) {
    NL = 1; NE = 4; table = calloc(1, sizeof(Expert *)); table[0] = calloc(4, sizeof(Expert));
    int8_t *kg[2], *kd[2];
    for (int e = 0; e < 2; e++) {
        kg[e] = malloc((size_t)2 * I * H); kd[e] = malloc((size_t)H * I);
        qmat_init(&table[0][e].gu, 2 * I, H, 11 + e, kg[e]); qmat_init(&table[0][e].dn, H, I, 977 + e, kd[e]);
    }
    int ntok = 3; float *x = malloc(sizeof(float) * ntok * H), *y = malloc(sizeof(float) * ntok * H);
    for (int i = 0; i < ntok * H; i++) x[i] = sinf(i * 0.37f) * (1.0f + (i % 7) * 0.1f);
    Pair pairs[] = {{0, 0, 0.7f}, {0, 1, 0.3f}, {1, 1, 1.0f}, {2, 0, 0.5f}, {2, 1, 0.5f}};
    run_layer(0, ntok, x, 5, pairs, y);
    double err = 0, ref2 = 0;
    for (int t = 0; t < ntok; t++) {
        float ref[H] = {0};
        for (int p = 0; p < 5; p++) if (pairs[p].tok == t) {
            int e = pairs[p].expert; uint8_t xu[H]; int32_t sxq[H / G]; float sx = quant_act(x + t * H, H, xu, sxq);
            float *g = malloc(2 * I * 4), h[I];
            for (int r = 0; r < 2 * I; r++) { double a = 0; for (int k = 0; k < H; k++) a += (double)((int)xu[k] - 128) * kg[e][(size_t)r * H + k] * table[0][e].gu.s[r * (H / G) + k / G]; g[r] = (float)(a * sx); }
            for (int i = 0; i < I; i++) h[i] = g[i] / (1.0f + expf(-g[i])) * g[I + i];
            uint8_t hq[I]; int32_t sh2[I / G]; float shs = quant_act(h, I, hq, sh2);
            for (int r = 0; r < H; r++) { double a = 0; for (int k = 0; k < I; k++) a += (double)((int)hq[k] - 128) * kd[e][(size_t)r * I + k] * table[0][e].dn.s[r * (I / G) + k / G]; ref[r] += pairs[p].w * (float)(a * shs); }
            free(g);
        }
        for (int r = 0; r < H; r++) { double d = y[t * H + r] - ref[r]; err += d * d; ref2 += (double)ref[r] * ref[r]; }
    }
    double rel = sqrt(err / ref2);
    printf("selftest multi-token batched vs scalar reference: rel L2 = %.3e %s\n", rel, rel < 1e-4 ? "SELFTEST OK" : "SELFTEST FAILED");
    return rel < 1e-4 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc < 3) { fprintf(stderr, "usage: %s serve <port> [layers] [experts] [resident] | serve-weights <port> <dir> | selftest\n", argv[0]); return 2; }
    int port = atoi(argv[2]);
    if (!strcmp(argv[1], "serve-weights")) {
        if (argc < 4 || load_weights(argv[3], 42, 288) <= 0) { fprintf(stderr, "no weights loaded\n"); return 1; }
    } else {
        int layers = argc > 3 ? atoi(argv[3]) : 42, experts = argc > 4 ? atoi(argv[4]) : 288, resident = argc > 5 ? atoi(argv[5]) : 288;
        fprintf(stderr, "building synthetic table: %d layers x %d resident experts (%.1f GB)\n", layers, resident,
                layers * (double)resident * (3.0 * H * I / 2 + 3.0 * H * I / G * 8) / 1e9);
        build_table(layers, experts, resident);
    }
    int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(srv, (void *)&a, sizeof a) || listen(srv, 4)) { perror("bind/listen"); return 1; }
    fprintf(stderr, "listening on %d\n", port);
    static float x[MAXTOK * H], y[MAXTOK * H]; static uint16_t xb[MAXTOK * H], yb[MAXTOK * H]; static Pair pairs[MAXPAIR];
    for (;;) {
        int c = accept(srv, NULL, NULL); if (c < 0) continue;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        for (;;) {
            uint32_t magic; uint16_t hdr[4];
            if (readn(c, &magic, 4) || magic != 0x31584648u || readn(c, hdr, 8)) break;
            int layer = hdr[0], ntok = hdr[1], np = hdr[2];
            if (ntok > MAXTOK || np > MAXPAIR || layer >= NL) break;
            struct { uint16_t tok, expert; float w; } __attribute__((packed)) wire[MAXPAIR];
            if (readn(c, wire, (size_t)np * 8) || readn(c, xb, (size_t)ntok * H * 2)) break;
            double t0 = now_us();
            for (int p = 0; p < np; p++) { pairs[p].tok = wire[p].tok; pairs[p].expert = wire[p].expert; pairs[p].w = wire[p].w; }
            for (int i = 0; i < ntok * H; i++) x[i] = bf16_to_f(xb[i]);
            int st = run_layer(layer, ntok, x, np, pairs, y);
            for (int i = 0; i < ntok * H; i++) yb[i] = f_to_bf16(y[i]);
            uint32_t us = (uint32_t)(now_us() - t0), rm = 0x31594648u; uint16_t rh[2] = {(uint16_t)ntok, (uint16_t)st};
            if (writen(c, &rm, 4) || writen(c, rh, 4) || writen(c, &us, 4) || writen(c, yb, (size_t)ntok * H * 2)) break;
        }
        close(c);
    }
}
