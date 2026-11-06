// v3: complete routed-expert FFN on CPU (AVX-512 VNNI), batch-1 decode.
//   y = down( silu(gate(x)) * up(x) ),  weights int4 with one fp32 scale per 128-weight group
//   (4.5 bpw incl. scale + per-group correction constant, comparable to Q4_K / RAWINT4).
// Activations are quantized per token to u8 with zero point 128 (absmax/127 scale).
// Integer identity per 128-group, with x_u = xq + 128 and w_u = w + 8 (nibble):
//   sum(xq * w) = dot(x_u, w_u) - 8 * S_xq - 128 * S_w - 128 * 128 * 8
// S_w (per row-group) is folded offline into c[r,g] = s_w * (128 * S_w + 131072); S_xq is per token.
// Mode "check" compares against a scalar float reference using the same quantized inputs.
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#define H 4096
#define I 2048
#define G 128
#define NEXP 64

typedef struct { uint8_t *q; float *s; float *c; int rows, cols; } QMat;   // q: rows*cols/2 bytes

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

static float quant_act(const float *x, int n, uint8_t *xu, int32_t *sxq) {   // returns scale
    float mx = 1e-12f; for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(x[i]));
    float s = mx / 127.0f, inv = 1.0f / s;
    for (int g = 0; g < n / G; g++) {
        int acc = 0;
        for (int i = g * G; i < (g + 1) * G; i++) { int q = (int)lrintf(x[i] * inv); acc += q; xu[i] = (uint8_t)(q + 128); }
        sxq[g] = acc;
    }
    return s;
}

// 4 output rows of a QMat against one quantized activation vector
static inline void rows4(const QMat *m, int r0, const uint8_t *xu, const int32_t *sxq, float sx, float *out) {
    const int ng = m->cols / G; const __m512i mk = _mm512_set1_epi8(0x0f);
    __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    float corr[4] = {0, 0, 0, 0};
    for (int g = 0; g < ng; g++) {
        __m512i xl = _mm512_loadu_si512(xu + g * G), xh = _mm512_loadu_si512(xu + g * G + 64);
        for (int r = 0; r < 4; r++) {
            const uint8_t *wq = m->q + ((size_t)(r0 + r) * m->cols + g * G) / 2;
            __m512i p = _mm512_loadu_si512(wq);
            __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), xl, _mm512_and_si512(p, mk));
            d = _mm512_dpbusd_epi32(d, xh, _mm512_and_si512(_mm512_srli_epi16(p, 4), mk));
            float sw = m->s[(r0 + r) * ng + g];
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_set1_ps(sw), acc[r]);
            corr[r] += sw * 8.0f * sxq[g] + m->c[(r0 + r) * ng + g];
        }
    }
    for (int r = 0; r < 4; r++) out[r] = sx * (_mm512_reduce_add_ps(acc[r]) - corr[r]);
}

typedef struct { QMat gu, dn; } Expert;   // gu: 2I x H (gate rows then up rows), dn: H x I

static void expert_fwd(const Expert *e, const uint8_t *xu, const int32_t *sxq, float sx,
                       float *gu_out, uint8_t *hu, int32_t *shq, float *y, float wgt) {
    #pragma omp for schedule(static)
    for (int r = 0; r < 2 * I; r += 4) rows4(&e->gu, r, xu, sxq, sx, gu_out + r);
    float sh;
    #pragma omp single
    {
        float *h = gu_out + 2 * I;     // scratch after gu_out
        for (int i = 0; i < I; i++) { float gt = gu_out[i]; h[i] = gt / (1.0f + expf(-gt)) * gu_out[I + i]; }
        sh = quant_act(h, I, hu, shq);
        gu_out[4 * I] = sh;
    }
    sh = gu_out[4 * I];
    #pragma omp for schedule(static)
    for (int r = 0; r < H; r += 4) { float o[4]; rows4(&e->dn, r, hu, shq, sh, o); for (int k = 0; k < 4; k++) y[r + k] += wgt * o[k]; }
}

// Layer-batched variant: all E experts' gate/up rows in one parallel loop, SiLU+requant in parallel
// across experts, then all down rows (3 barriers per layer instead of 3 per expert).
static void layer_fwd(Expert **es, int E, const uint8_t *xu, const int32_t *sxq, float sx,
                      float *gu_all, uint8_t *hu_all, int32_t *shq_all, float *sh_all, float *ypart, const float *wgt) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static) collapse(2)
        for (int e = 0; e < E; e++) for (int r = 0; r < 2 * I; r += 4) rows4(&es[e]->gu, r, xu, sxq, sx, gu_all + (size_t)e * 2 * I + r);
        #pragma omp for schedule(static)
        for (int e = 0; e < E; e++) {
            float h[I]; const float *g = gu_all + (size_t)e * 2 * I;
            for (int i = 0; i < I; i++) h[i] = g[i] / (1.0f + expf(-g[i])) * g[I + i];
            sh_all[e] = quant_act(h, I, hu_all + (size_t)e * I, shq_all + e * (I / G));
        }
        #pragma omp for schedule(static)
        for (int r = 0; r < H; r += 4) {
            float o[4], acc[4] = {0, 0, 0, 0};
            for (int e = 0; e < E; e++) { rows4(&es[e]->dn, r, hu_all + (size_t)e * I, shq_all + e * (I / G), sh_all[e], o);
                for (int k = 0; k < 4; k++) acc[k] += wgt[e] * o[k]; }
            for (int k = 0; k < 4; k++) ypart[r + k] = acc[k];
        }
    }
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "bench";
    int E = argc > 2 ? atoi(argv[2]) : 8;
    int npool = strcmp(mode, "check") ? NEXP : 1;
    Expert *ex = malloc(sizeof(Expert) * npool);
    int8_t *kg = NULL, *kd = NULL;
    if (!strcmp(mode, "check")) { kg = malloc((size_t)2 * I * H); kd = malloc((size_t)H * I); }
    for (int i = 0; i < npool; i++) { qmat_init(&ex[i].gu, 2 * I, H, 11 + i, kg); qmat_init(&ex[i].dn, H, I, 977 + i, kd); }
    float *x = aligned_alloc(64, H * 4), *y = aligned_alloc(64, H * 4), *scr = aligned_alloc(64, (4 * I + 16) * 4);
    uint8_t *xu = aligned_alloc(64, H), *hu = aligned_alloc(64, I); int32_t sxq[H / G], shq[I / G];
    for (int i = 0; i < H; i++) x[i] = sinf(i * 0.37f) * (1.0f + (i % 7) * 0.1f);
    float sx = quant_act(x, H, xu, sxq);
    if (!strcmp(mode, "check")) {
        memset(y, 0, H * 4);
        #pragma omp parallel
        expert_fwd(&ex[0], xu, sxq, sx, scr, hu, shq, y, 1.0f);
        // scalar reference using identical quantized x and identical h quantization
        float *gu = malloc(2 * I * 4), *h = malloc(I * 4); double err = 0, ref2 = 0;
        for (int r = 0; r < 2 * I; r++) { double a = 0; for (int k = 0; k < H; k++) a += (double)((int)xu[k] - 128) * kg[(size_t)r * H + k] * ex[0].gu.s[r * (H / G) + k / G]; gu[r] = (float)(a * sx); }
        for (int i = 0; i < I; i++) h[i] = gu[i] / (1.0f + expf(-gu[i])) * gu[I + i];
        uint8_t hq[I]; int32_t sh2[I / G]; float shs = quant_act(h, I, hq, sh2);
        for (int r = 0; r < H; r++) { double a = 0; for (int k = 0; k < I; k++) a += (double)((int)hq[k] - 128) * kd[(size_t)r * I + k] * ex[0].dn.s[r * (I / G) + k / G];
            double ref = a * shs; err += (y[r] - ref) * (y[r] - ref); ref2 += ref * ref; }
        printf("check: relative L2 error vs scalar reference = %.3e  %s\n", sqrt(err / ref2), sqrt(err / ref2) < 1e-4 ? "CHECK OK" : "CHECK FAILED");
        return 0;
    }
    int batched = !strcmp(mode, "bench2");
    float *gu_all = aligned_alloc(64, (size_t)64 * 2 * I * 4); uint8_t *hu_all = aligned_alloc(64, (size_t)64 * I);
    int32_t *shq_all = aligned_alloc(64, (size_t)64 * (I / G) * 4); float sh_all[64], wg[64]; Expert *es[64];
    for (int k = 0; k < 64; k++) wg[k] = 0.125f;
    int iters = 300; double t0 = 0;
    for (int it = -30; it < iters; it++) {
        if (it == 0) t0 = omp_get_wtime();
        memset(y, 0, H * 4);
        if (batched) {
            for (int k = 0; k < E; k++) es[k] = &ex[((it + 1000) * E + k) % NEXP];
            layer_fwd(es, E, xu, sxq, sx, gu_all, hu_all, shq_all, sh_all, y, wg);
        } else {
            #pragma omp parallel
            for (int k = 0; k < E; k++) expert_fwd(&ex[((it + 1000) * E + k) % NEXP], xu, sxq, sx, scr, hu, shq, y, 0.125f);
        }
    }
    double dt = (omp_get_wtime() - t0) / iters;
    double bytes = (double)E * ((size_t)3 * H * I / 2 + (size_t)3 * H * I / G * 8);
    printf("v3%s full expert int4/g128 E=%d threads=%d: %.3f ms/layer, %.1f us/expert, %.0f GB/s (weights+scales)\n",
           batched ? "-batched" : "", E, omp_get_max_threads(), dt * 1e3, dt * 1e6 / E, bytes / dt / 1e9);
    return 0;
}
