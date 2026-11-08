// HeteroFlash runtime-v1: Spark-side bench of the per-layer heterogeneous expert exchange (expert_server.c).
// Emulates one decode step / speculative verify step: 42 sequential per-layer requests. Per layer, T tokens each
// select 8 experts drawn from a window union of U distinct experts (union sizes from the M1 traces); the CPU owns a
// fraction f of the union and receives only the (token, expert, weight) pairs of its experts plus T x 4096 bf16
// activations, and replies with the summed partial output per token.
//
// Usage: layer_exchange_bench bench <host> <port> <table_layers e.g. 17,27,41> <mode 0 plain|1 busy-poll|3 busy-poll+heartbeat> [steps=40]
//        layer_exchange_bench check <host> <port> <table_layers> [requests=20]
//   table_layers are expert_server table indices (absolute decoder layer - 3) that the server has loaded.
//   check: sends identical requests on a busy-poll and a plain connection and compares the replies bit for bit.
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
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
#define I 2048
#define G 128
#define NEXP 288
#define MAXT 16
static const double EXPERT_BYTES = (2.0 * I * H / 2 + 2.0 * I * (H / G) * 8) + ((double)H * I / 2 + (double)H * (I / G) * 8);

static int busy = 0;
static int readn(int fd, void *b, size_t n) {
    size_t o = 0;
    while (o < n) {
        ssize_t k = recv(fd, (char *)b + o, n - o, busy ? MSG_DONTWAIT : 0);
        if (k > 0) { o += k; continue; }
        if (k == 0) return -1;
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
        return -1;
    }
    if (busy) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one); }
    return 0;
}
static int writen(int fd, const void *b, size_t n) {
    size_t o = 0;
    while (o < n) { ssize_t k = send(fd, (const char *)b + o, n - o, MSG_NOSIGNAL); if (k < 0 && (errno == EAGAIN || errno == EINTR)) continue; if (k <= 0) return -1; o += k; }
    return 0;
}
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double pct(double *v, int n, double q) { qsort(v, n, sizeof(double), cmpd); return v[(int)(q * (n - 1))]; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static unsigned rs = 12345;
static unsigned rnd(void) { rs = rs * 1103515245u + 12345u; return rs >> 8; }
static float rndn(void) { float u1 = (rnd() % 1000000 + 1) / 1000001.0f, u2 = (rnd() % 1000000) / 1000000.0f; return sqrtf(-2 * logf(u1)) * cosf(6.2831853f * u2); }

static int connect_to(const char *host, int port, int want_busy) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1, bp = 50;
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, host, &a.sin_addr);
    if (connect(fd, (void *)&a, sizeof a)) { perror("connect"); exit(1); }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (want_busy) { setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one); setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &bp, sizeof bp); }
    return fd;
}

typedef struct __attribute__((packed)) { uint16_t tok, expert; float w; } Pair;
typedef struct { int layer, ntok, np, ncpu; Pair pr[MAXT * 8]; uint16_t xb[MAXT * H]; } Req;

// T tokens x 8 experts from a union of U experts; CPU owns round(f*U) of the union.
static void make_req(Req *r, int layer, int T, int U, double f) {
    int pool[NEXP], used[NEXP] = {0}, n = 0;
    while (n < U) { int e = rnd() % NEXP; if (!used[e]) { used[e] = 1; pool[n++] = e; } }
    int sel[MAXT][8], cnt[MAXT] = {0};
    for (int k = 0; k < U; k++) { int t = k % T; if (cnt[t] < 8) sel[t][cnt[t]++] = pool[k]; }   // cover the union
    for (int t = 0; t < T; t++) while (cnt[t] < 8) {
        int e = pool[rnd() % U], dup = 0;
        for (int j = 0; j < cnt[t]; j++) if (sel[t][j] == e) dup = 1;
        if (!dup) sel[t][cnt[t]++] = e;
    }
    int ncpu = (int)lrint(f * U), cpu[NEXP] = {0};
    for (int k = 0; k < ncpu; k++) cpu[pool[k]] = 1;   // pool order is random, so the first ncpu are a random subset
    r->layer = layer; r->ntok = T; r->np = 0; r->ncpu = ncpu;
    for (int t = 0; t < T; t++) {
        float w[8], s = 0; for (int j = 0; j < 8; j++) { w[j] = 0.2f + (rnd() % 1000) / 1000.0f; s += w[j]; }
        for (int j = 0; j < 8; j++) if (cpu[sel[t][j]]) { r->pr[r->np].tok = t; r->pr[r->np].expert = sel[t][j]; r->pr[r->np].w = w[j] / s; r->np++; }
    }
    for (int i = 0; i < T * H; i++) r->xb[i] = f2bf(0.05f * rndn());
}

// returns rtt us; server compute us in *srv; reply bytes in yb
static double send_req(int fd, const Req *r, int flag_busy, uint32_t *srv, uint16_t *yb) {
    uint32_t magic = 0x31584648u; uint16_t hdr[4] = {(uint16_t)r->layer, (uint16_t)r->ntok, (uint16_t)r->np, (uint16_t)flag_busy};
    double t0 = now_us();
    if (writen(fd, &magic, 4) || writen(fd, hdr, 8) || writen(fd, r->pr, (size_t)r->np * 8) || writen(fd, r->xb, (size_t)r->ntok * H * 2)) { perror("write"); exit(1); }
    uint32_t rm; uint16_t rh[2]; unsigned char c0;
    do { if (readn(fd, &c0, 1)) { perror("read"); exit(1); } } while (c0 == 0);   // skip heartbeat bytes
    if (readn(fd, (char *)&rm + 1, 3)) { perror("read"); exit(1); } ((unsigned char *)&rm)[0] = c0;
    if ( readn(fd, rh, 4) || readn(fd, srv, 4) || readn(fd, yb, (size_t)r->ntok * H * 2)) { perror("read"); exit(1); }
    double t1 = now_us();
    if (rm != 0x31594648u || rh[1]) { fprintf(stderr, "bad reply magic %x status %d (layer %d not loaded?)\n", rm, rh[1], r->layer); exit(1); }
    return t1 - t0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: see header\n"); return 2; }
    const char *host = argv[2]; int port = atoi(argv[3]);
    int layers[64], nl = 0; { char buf[256]; snprintf(buf, sizeof buf, "%s", argv[4]); for (char *p = strtok(buf, ","); p; p = strtok(NULL, ",")) layers[nl++] = atoi(p); }
    static uint16_t ya[MAXT * H], yb2[MAXT * H];
    if (!strcmp(argv[1], "check")) {
        int n = argc > 5 ? atoi(argv[5]) : 20;
        // the server serves one connection at a time: busy-poll pass first (replies kept), then a plain pass
        if (n > 64) n = 64;
        static Req reqs[64]; static uint16_t yk[64][MAXT * H];
        int bad = 0; uint32_t s1;
        int Ts[] = {1, 4, 8}, Us[] = {8, 19, 26};
        for (int i = 0; i < n; i++) { int k = i % 3; make_req(&reqs[i], layers[i % nl], Ts[k], Us[k], 1.0); }
        int fb = connect_to(host, port, 1); busy = 1;
        for (int i = 0; i < n; i++) send_req(fb, &reqs[i], 3, &s1, yk[i]);   // busy-poll + heartbeat
        close(fb);
        int fp = connect_to(host, port, 0); busy = 0;
        for (int i = 0; i < n; i++) { send_req(fp, &reqs[i], 0, &s1, yb2); if (memcmp(yk[i], yb2, (size_t)reqs[i].ntok * H * 2)) bad++; }
        close(fp);
        printf("check: %d requests, busy-poll+heartbeat vs plain replies identical: %s (%d differ)\n", n, bad ? "NO" : "YES", bad);
        return bad != 0;
    }
    int mode = atoi(argv[5]); busy = mode & 1; int steps = argc > 6 ? atoi(argv[6]) : 40;
    int fd = connect_to(host, port, busy);
    struct { const char *name; int T, U; } wl[] = {{"decode", 1, 8}, {"verify4", 4, 19}, {"verify4", 4, 23}, {"verify8", 8, 26}};
    double fs[] = {0.0, 0.5, 0.85, 1.0};   // f = 0: no CPU experts, pure protocol + activation conversion overhead
    printf("mode=%d (busy=%d heartbeat=%d) steps=%d layers_per_step=42 expert_bytes=%.2fMB\n", mode, busy, (mode >> 1) & 1, steps, EXPERT_BYTES / 1e6);
    printf("workload T U f ncpu | rtt_us p50 p99 | server_us p50 | net_us p50 | eff_GBs (distinct expert bytes / server time) | step_ms(42 layers) p50\n");
    for (unsigned w = 0; w < sizeof wl / sizeof wl[0]; w++) for (int fi = 0; fi < 4; fi++) {
        int N = steps * 42; double *rtt = malloc(N * 8), *sv = malloc(N * 8), *net = malloc(N * 8), *bw = malloc(N * 8), *st = malloc(steps * 8);
        int nc = 0, R = 168; static Req pre[168];   // pre-generated so request synthesis is outside the timed path
        for (int k = 0; k < R; k++) make_req(&pre[k], layers[(k % 42) % nl], wl[w].T, wl[w].U, fs[fi]);
        for (int s = -3; s < steps; s++) {
            double t0 = now_us();
            for (int l = 0; l < 42; l++) {
                const Req *q = &pre[((s + 3) * 42 + l) % R];
                uint32_t srv; double r = send_req(fd, q, mode, &srv, ya);
                if (s >= 0) { int i = s * 42 + l; rtt[i] = r; sv[i] = srv; net[i] = r - srv; bw[i] = srv > 0 ? q->ncpu * EXPERT_BYTES / (srv * 1e-6) / 1e9 : 0; nc = q->ncpu; }
            }
            if (s >= 0) st[s] = (now_us() - t0) / 1e3;
        }
        printf("%-8s %2d %2d %.2f %3d | %7.0f %7.0f | %7.0f | %6.0f | %6.1f | %7.2f\n", wl[w].name, wl[w].T, wl[w].U, fs[fi], nc,
               pct(rtt, N, .5), pct(rtt, N, .99), pct(sv, N, .5), pct(net, N, .5), pct(bw, N, .5), pct(st, steps, .5));
        fflush(stdout);
        free(rtt); free(sv); free(net); free(bw); free(st);
    }
    return 0;
}
