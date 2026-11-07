// Spark-side benchmark client for runtime/expert_server.c.
// Emulates the remote-expert calls of one speculative verify step: 42 sequential
// per-layer requests, each with `ntok` tokens and `nexp` distinct remote experts per
// layer (spread across the tokens like a real window union). Reports per-layer round
// trip, server compute time, and their difference (network + serialization).
// Usage: expert_client_bench <host> <port> <resident_per_layer> [steps=30]
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#define H 4096
static int readn(int fd, void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = read(fd, (char *)b + o, n - o); if (k <= 0) return -1; o += k; } return 0; }
static int writen(int fd, const void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = write(fd, (const char *)b + o, n - o); if (k <= 0) return -1; o += k; } return 0; }
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
static double pct(double *v, int n, double q) { qsort(v, n, sizeof(double), cmp); return v[(int)(q * (n - 1))]; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s host port resident_per_layer [steps]\n", argv[0]); return 2; }
    int resident = atoi(argv[3]), steps = argc > 4 ? atoi(argv[4]) : 30;
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(atoi(argv[2]))};
    inet_pton(AF_INET, argv[1], &a.sin_addr);
    if (connect(fd, (void *)&a, sizeof a)) { perror("connect"); return 1; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    static uint16_t xb[16 * H], yb[16 * H];
    for (int i = 0; i < 16 * H; i++) xb[i] = 0x3c00 + (i % 97);          // bf16 values ~0.0078..
    int toks[] = {1, 4, 6, 8}, exps[] = {1, 2, 4, 8, 16};
    srand(1);
    printf("ntok nexp | layer_rtt_us p50 p95 | server_us p50 | net_us p50 | per_step_ms(42 layers) p50\n");
    for (int ti = 0; ti < 4; ti++) for (int ei = 0; ei < 5; ei++) {
        int ntok = toks[ti], nexp = exps[ei];
        if (nexp > resident) continue;
        int nlay = 42, N = steps * nlay;
        double *rtt = malloc(sizeof(double) * N), *srv = malloc(sizeof(double) * N), *net = malloc(sizeof(double) * N), *stp = malloc(sizeof(double) * steps);
        for (int s = -3; s < steps; s++) {
            double st0 = now_us();
            for (int l = 0; l < nlay; l++) {
                // nexp distinct experts; every expert gets >=1 token, total pairs <= ntok*8
                struct { uint16_t tok, expert; float w; } __attribute__((packed)) pr[128];
                int np = 0;
                for (int e = 0; e < nexp; e++) {
                    int ex = (rand() % resident);
                    int per = 1 + (ntok > 1 ? rand() % ntok : 0);
                    for (int k = 0; k < per && np < 128; k++) { pr[np].tok = (uint16_t)((e + k) % ntok); pr[np].expert = (uint16_t)ex; pr[np].w = 0.1f; np++; }
                }
                // dedupe (tok,expert) pairs
                int m = 0;
                for (int i = 0; i < np; i++) { int dup = 0; for (int j = 0; j < m; j++) if (pr[j].tok == pr[i].tok && pr[j].expert == pr[i].expert) dup = 1; if (!dup) pr[m++] = pr[i]; }
                np = m;
                uint32_t magic = 0x31584648u; uint16_t hdr[4] = {(uint16_t)l, (uint16_t)ntok, (uint16_t)np, 0};
                double t0 = now_us();
                if (writen(fd, &magic, 4) || writen(fd, hdr, 8) || writen(fd, pr, (size_t)np * 8) || writen(fd, xb, (size_t)ntok * H * 2)) { perror("write"); return 1; }
                uint32_t rm, us; uint16_t rh[2];
                if (readn(fd, &rm, 4) || readn(fd, rh, 4) || readn(fd, &us, 4) || readn(fd, yb, (size_t)ntok * H * 2)) { perror("read"); return 1; }
                double t1 = now_us();
                if (rh[1]) { fprintf(stderr, "server status %d\n", rh[1]); return 1; }
                if (s >= 0) { int i = s * nlay + l; rtt[i] = t1 - t0; srv[i] = us; net[i] = rtt[i] - us; }
            }
            if (s >= 0) stp[s] = (now_us() - st0) / 1e3;
        }
        printf("%4d %4d | %8.0f %8.0f | %8.0f | %8.0f | %8.2f\n", ntok, nexp, pct(rtt, N, .5), pct(rtt, N, .95), pct(srv, N, .5), pct(net, N, .5), pct(stp, steps, .5));
        fflush(stdout);
        free(rtt); free(srv); free(net); free(stp);
    }
    close(fd);
    return 0;
}
