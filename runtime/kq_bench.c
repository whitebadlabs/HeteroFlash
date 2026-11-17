// Synthetic per-format CPU expert bench (2026-11-17): 11 experts x [gate,up 2048x4096 + down 4096x2048] of random
// weights quantized in memory to the given ggml type (no imatrix), timed with the same vec_dot loop structure as
// expert_server_gguf run_layer (speed only; numbers are not a quality statement).  kq_bench <type> <ntok> <reps>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"
#define H 4096
#define I 2048
#define NE 11
int main(int argc, char **argv) {
    ggml_cpu_init();
    enum ggml_type ty = (enum ggml_type)atoi(argv[1]); int nt = atoi(argv[2]), reps = atoi(argv[3]);
    const struct ggml_type_traits_cpu *tr = ggml_get_type_traits_cpu(ty);
    const struct ggml_type_traits_cpu *qv = ggml_get_type_traits_cpu(tr->vec_dot_type);
    size_t rg = ggml_row_size(ty, H), rd = ggml_row_size(ty, I);
    int nsets = 24;   // rotate over 24 sets of 11 experts so the weights stream from DRAM (> L3)
    size_t per = (size_t)2 * I * rg + (size_t)H * rd;
    uint8_t *W = malloc(per * NE * nsets);
    float *tmp = malloc(sizeof(float) * I * H);
    for (size_t i = 0; i < (size_t)I * H; i++) tmp[i] = (float)((rand() / (double)RAND_MAX) - 0.5) * 0.05f;
    #pragma omp parallel for
    for (int e = 0; e < NE * nsets; e++) {
        uint8_t *b = W + per * e;
        ggml_quantize_chunk(ty, tmp, b, 0, I, H, NULL);
        ggml_quantize_chunk(ty, tmp, b + (size_t)I * rg, 0, I, H, NULL);
        ggml_quantize_chunk(ty, tmp, b + (size_t)2 * I * rg, 0, H, I, NULL);
    }
    static float x[16][H], g[NE][16][I], u[NE][16][I], y[16][H]; static uint8_t xq[16][H * 2], hq[NE][16][I * 2];
    for (int t = 0; t < nt; t++) for (int i = 0; i < H; i++) x[t][i] = sinf(i * 0.37f + t);
    double sum = 0, best = 1e30;
    for (int r = 0; r < reps + 3; r++) {
        uint8_t *base = W + per * NE * (r % nsets);
        double t0 = omp_get_wtime();
        for (int t = 0; t < nt; t++) qv->from_float(x[t], xq[t], H);
        #pragma omp parallel
        {
            #pragma omp for schedule(static) collapse(2)
            for (int e = 0; e < NE; e++) for (int rr = 0; rr < I; rr++) {
                const uint8_t *wg = base + per * e + (size_t)rr * rg, *wu = base + per * e + (size_t)(I + rr) * rg;
                for (int k = 0; k < nt; k++) { tr->vec_dot(H, &g[e][k][rr], 0, wg, 0, xq[k], 0, 1); tr->vec_dot(H, &u[e][k][rr], 0, wu, 0, xq[k], 0, 1); }
            }
            #pragma omp for schedule(static) collapse(2)
            for (int e = 0; e < NE; e++) for (int k = 0; k < 16; k++) {
                if (k >= nt) continue;
                float h[I]; for (int i = 0; i < I; i++) { float gv = fminf(g[e][k][i], 10.f); h[i] = gv / (1.f + expf(-gv)) * u[e][k][i]; }
                qv->from_float(h, hq[e][k], I);
            }
            #pragma omp for schedule(static)
            for (int rr = 0; rr < H; rr++) for (int e = 0; e < NE; e++) {
                const uint8_t *wd = base + per * e + (size_t)2 * I * rg + (size_t)rr * rd;
                for (int k = 0; k < nt; k++) { float s; tr->vec_dot(I, &s, 0, wd, 0, hq[e][k], 0, 1); y[k][rr] += s; }
            }
        }
        double dt = (omp_get_wtime() - t0) * 1e6;
        if (r >= 3) { sum += dt; if (dt < best) best = dt; }
    }
    printf("{\"type\": \"%s\", \"ntok\": %d, \"MB_per_expert\": %.2f, \"mean_us\": %.1f, \"best_us\": %.1f}\n", ggml_type_name(ty), nt, per / 1e6, sum / reps, best);
    return 0;
}
