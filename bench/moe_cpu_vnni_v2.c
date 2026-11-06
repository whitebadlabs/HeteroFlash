// v2 CPU MoE expert kernel: AVX-512 VNNI, batch-1 GEMV, int8 or int4 weights.
// Changes vs moe_cpu_vnni.c: 4-row register blocking (activations loaded once per
// 4 rows), int4 stored as unsigned nibbles with the zero point (8) folded into a
// per-row correction (-8 * sum(x)), no per-weight subtract.
// Layout per expert: gate|up rows [2*I][H], down rows [H][I]; int4 packs 128
// weights per 64 bytes: low nibbles = elements k..k+63, high nibbles = k+64..k+127.
// Weights are cycled through a pool larger than LLC so reads come from DRAM.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#define H 4096
#define I 2048
#define NEXP 96

static inline void dot4_s8(const uint8_t *x, const int8_t *w, size_t ld, int n, int32_t out[4]) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < n; k += 64) {
        __m512i xv = _mm512_loadu_si512(x + k);
        a0 = _mm512_dpbusd_epi32(a0, xv, _mm512_loadu_si512(w + k));
        a1 = _mm512_dpbusd_epi32(a1, xv, _mm512_loadu_si512(w + ld + k));
        a2 = _mm512_dpbusd_epi32(a2, xv, _mm512_loadu_si512(w + 2 * ld + k));
        a3 = _mm512_dpbusd_epi32(a3, xv, _mm512_loadu_si512(w + 3 * ld + k));
    }
    out[0] = _mm512_reduce_add_epi32(a0); out[1] = _mm512_reduce_add_epi32(a1);
    out[2] = _mm512_reduce_add_epi32(a2); out[3] = _mm512_reduce_add_epi32(a3);
}

static inline void dot4_u4(const uint8_t *x, const uint8_t *w, size_t ldb, int n, int32_t xsum, int32_t out[4]) {
    const __m512i m = _mm512_set1_epi8(0x0f);
    __m512i a[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
    for (int k = 0; k < n; k += 128) {
        __m512i xl = _mm512_loadu_si512(x + k), xh = _mm512_loadu_si512(x + k + 64);
        for (int r = 0; r < 4; r++) {
            __m512i p = _mm512_loadu_si512(w + r * ldb + k / 2);
            a[r] = _mm512_dpbusd_epi32(a[r], xl, _mm512_and_si512(p, m));
            a[r] = _mm512_dpbusd_epi32(a[r], xh, _mm512_and_si512(_mm512_srli_epi16(p, 4), m));
        }
    }
    for (int r = 0; r < 4; r++) out[r] = _mm512_reduce_add_epi32(a[r]) - 8 * xsum;
}

static int32_t sum_u8(const uint8_t *x, int n) { int32_t s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <bits 8|4> <experts_per_token> <tokens>\n", argv[0]); return 2; }
    int bits = atoi(argv[1]), E = atoi(argv[2]), T = atoi(argv[3]);
    size_t per = (size_t)3 * H * I * bits / 8;
    uint8_t *W = aligned_alloc(64, per * NEXP); memset(W, 0x5a, per * NEXP);
    uint8_t *x = aligned_alloc(64, (size_t)H * T); memset(x, 3, (size_t)H * T);
    int32_t *y = aligned_alloc(64, (size_t)4 * H * 64); memset(y, 0, (size_t)4 * H * 64);
    int distinct = E * T < NEXP ? E * T : NEXP, tok_per = (E * T + distinct - 1) / distinct;
    int32_t xs_h = sum_u8(x, H), xs_i = sum_u8(x, I);
    int iters = 200; double t0 = 0;
    for (int it = -20; it < iters; it++) {
        if (it == 0) t0 = omp_get_wtime();
        int base = ((it + 1000) * distinct) % NEXP;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int e = 0; e < distinct; e++) for (int r = 0; r < 2 * I; r += 4) {
            const uint8_t *we = W + per * ((base + e) % NEXP);
            for (int t = 0; t < tok_per; t++) {
                int32_t o[4];
                if (bits == 8) dot4_s8(x + (size_t)t * H, (const int8_t *)we + (size_t)r * H, H, H, o);
                else dot4_u4(x + (size_t)t * H, we + (size_t)r * H / 2, H / 2, H, xs_h, o);
                y[(r % (4 * H)) + (e & 63)] += o[0] + o[1] + o[2] + o[3];
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int e = 0; e < distinct; e++) for (int r = 0; r < H; r += 4) {
            const uint8_t *we = W + per * ((base + e) % NEXP) + (size_t)2 * I * H * bits / 8;
            for (int t = 0; t < tok_per; t++) {
                int32_t o[4];
                if (bits == 8) dot4_s8(x + (size_t)t * I, (const int8_t *)we + (size_t)r * I, I, I, o);
                else dot4_u4(x + (size_t)t * I, we + (size_t)r * I / 2, I / 2, I, xs_i, o);
                y[(r % (4 * H)) + (e & 63)] += o[0] + o[1] + o[2] + o[3];
            }
        }
    }
    double dt = (omp_get_wtime() - t0) / iters, gb = (double)per * distinct / 1e9;
    printf("v2 int%d experts/tok=%d tokens=%d distinct=%d threads=%d: %.3f ms/layer, %.1f us/expert, %.0f GB/s weights\n",
           bits, E, T, distinct, omp_get_max_threads(), dt * 1e3, dt * 1e6 / distinct, gb / dt);
    return y[7] == 123456789;
}
