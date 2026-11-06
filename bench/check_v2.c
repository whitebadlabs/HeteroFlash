// Correctness check for moe_cpu_vnni_v2.c dot kernels vs scalar reference.
#define main bench_main
#include "moe_cpu_vnni_v2.c"
#undef main
int main(void) {
    srand(7); int n = 4096, bad = 0;
    uint8_t *x = aligned_alloc(64, n), *w4 = aligned_alloc(64, 4 * n / 2); int8_t *w8 = aligned_alloc(64, 4 * n);
    int8_t q[4][4096];
    for (int i = 0; i < n; i++) x[i] = rand() & 255;
    for (int r = 0; r < 4; r++) for (int k = 0; k < n; k++) { q[r][k] = (rand() % 16) - 8; w8[r * n + k] = q[r][k]; }
    for (int r = 0; r < 4; r++) for (int k = 0; k < n; k += 128) for (int j = 0; j < 64; j++)
        w4[r * n / 2 + k / 2 + j] = (uint8_t)((q[r][k + j] + 8) | ((q[r][k + 64 + j] + 8) << 4));
    int32_t o4[4], o8[4]; dot4_u4(x, w4, n / 2, n, sum_u8(x, n), o4); dot4_s8(x, w8, n, n, o8);
    for (int r = 0; r < 4; r++) { long ref = 0; for (int k = 0; k < n; k++) ref += (long)x[k] * q[r][k];
        if (ref != o4[r] || ref != o8[r]) bad++; printf("row %d ref %ld int4 %d int8 %d\n", r, ref, o4[r], o8[r]); }
    puts(bad ? "CHECK FAILED" : "CHECK OK"); return bad;
}
