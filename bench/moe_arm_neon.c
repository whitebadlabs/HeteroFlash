// Spark Arm (Cortex-X925/A725) expert GEMV with NEON SDOT, int8 or int4 (unsigned nibble + zero-point fold),
// 4-row blocking; gate+up 4096 -> 2x2048 and down 2048 -> 4096 per expert; weights cycled over a pool > caches.
// Usage: moe_arm <bits 8|4> <experts> [check]
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#define H 4096
#define I 2048
#define NEXP 48

static inline void dot4_s8(const int8_t *x, const int8_t *w, size_t ld, int n, int32_t o[4]) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < n; k += 16) {
        int8x16_t xv = vld1q_s8(x + k);
        a0 = vdotq_s32(a0, xv, vld1q_s8(w + k));          a1 = vdotq_s32(a1, xv, vld1q_s8(w + ld + k));
        a2 = vdotq_s32(a2, xv, vld1q_s8(w + 2 * ld + k)); a3 = vdotq_s32(a3, xv, vld1q_s8(w + 3 * ld + k));
    }
    o[0] = vaddvq_s32(a0); o[1] = vaddvq_s32(a1); o[2] = vaddvq_s32(a2); o[3] = vaddvq_s32(a3);
}
// int4: 32 weights per 16 bytes (low nibbles = k..k+15, high = k+16..k+31), stored unsigned (w+8); x signed int8.
static inline void dot4_u4(const int8_t *x, const uint8_t *w, size_t ldb, int n, int32_t xsum, int32_t o[4]) {
    const uint8x16_t m = vdupq_n_u8(0x0f);
    int32x4_t a[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    for (int k = 0; k < n; k += 32) {
        int8x16_t xl = vld1q_s8(x + k), xh = vld1q_s8(x + k + 16);
        for (int r = 0; r < 4; r++) {
            uint8x16_t p = vld1q_u8(w + r * ldb + k / 2);
            a[r] = vdotq_s32(a[r], xl, vreinterpretq_s8_u8(vandq_u8(p, m)));
            a[r] = vdotq_s32(a[r], xh, vreinterpretq_s8_u8(vshrq_n_u8(p, 4)));
        }
    }
    for (int r = 0; r < 4; r++) o[r] = vaddvq_s32(a[r]) - 8 * xsum;
}

int main(int argc, char **argv) {
    int bits = atoi(argv[1]), E = atoi(argv[2]);
    size_t per = (size_t)3 * H * I * bits / 8;
    uint8_t *W = aligned_alloc(64, per * NEXP);
    for (size_t i = 0; i < per * NEXP; i++) W[i] = (uint8_t)(i * 2654435761u >> 24);
    int8_t *x = aligned_alloc(64, H); int32_t xs_h = 0, xs_i = 0;
    for (int i = 0; i < H; i++) { x[i] = (int8_t)((i * 37) % 255 - 127); xs_h += x[i]; if (i < I) xs_i += x[i]; }
    if (argc > 3) {  // correctness check of dot kernels against scalar
        int bad = 0;
        for (int r = 0; r < 4; r++) {
            int32_t o[4]; long ref = 0;
            if (bits == 8) { dot4_s8(x, (int8_t *)W, H, H, o); for (int k = 0; k < H; k++) ref += (long)x[k] * ((int8_t *)W)[(size_t)r * H + k]; }
            else { dot4_u4(x, W, H / 2, H, xs_h, o);
                   for (int k = 0; k < H; k += 32) for (int j = 0; j < 16; j++) { uint8_t p = W[(size_t)r * H / 2 + k / 2 + j];
                       ref += (long)x[k + j] * ((p & 15) - 8) + (long)x[k + 16 + j] * ((p >> 4) - 8); } }
            bad += (ref != o[r]);
        }
        printf("check int%d: %s\n", bits, bad ? "FAILED" : "OK");
        return bad;
    }
    int32_t *y = calloc(4 * H * 64, 4);
    int iters = 100; double t0 = 0;
    for (int it = -10; it < iters; it++) {
        if (it == 0) t0 = omp_get_wtime();
        int base = ((it + 1000) * E) % NEXP;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int e = 0; e < E; e++) for (int r = 0; r < 2 * I; r += 4) {
            const uint8_t *we = W + per * ((base + e) % NEXP); int32_t o[4];
            if (bits == 8) dot4_s8(x, (const int8_t *)we + (size_t)r * H, H, H, o); else dot4_u4(x, we + (size_t)r * H / 2, H / 2, H, xs_h, o);
            y[r + (e & 63)] += o[0] + o[3];
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int e = 0; e < E; e++) for (int r = 0; r < H; r += 4) {
            const uint8_t *we = W + per * ((base + e) % NEXP) + (size_t)2 * I * H * bits / 8; int32_t o[4];
            if (bits == 8) dot4_s8(x, (const int8_t *)we + (size_t)r * I, I, I, o); else dot4_u4(x, we + (size_t)r * I / 2, I / 2, I, xs_i, o);
            y[r + (e & 63)] += o[1] + o[2];
        }
    }
    double dt = (omp_get_wtime() - t0) / iters;
    printf("arm int%d E=%d threads=%d: %.3f ms/layer, %.1f us/expert, %.0f GB/s\n", bits, E, omp_get_max_threads(),
           dt * 1e3, dt * 1e6 / E, per * (double)E / dt / 1e9);
    return y[3] == 12345;
}
