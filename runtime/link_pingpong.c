// Direct-link round-trip microbenchmark: request of REQ bytes, reply of REP bytes, TCP_NODELAY, optional busy polling.
//   server: link_pingpong s <bind_ip> <port> <req> <rep> [busy] [delay_us] [heartbeat_us]
//   delay_us: server spins this long before replying (emulates expert compute between request and reply).
//   busy=3: non-blocking spin recv without SO_BUSY_POLL.
//   busy=4: blocking recv with SO_BUSY_POLL 5000 us (one long kernel busy loop per wait).
//   busy=2: also SO_PREFER_BUSY_POLL + SO_BUSY_POLL_BUDGET (keeps the NAPI context in busy-poll mode).
//   client: link_pingpong c <server_ip> <port> <req> <rep> <iters> [busy]
// busy=1: SO_BUSY_POLL 50 us + non-blocking spin on recv (no sleep in the kernel wait path).
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#ifndef SO_PREFER_BUSY_POLL
#define SO_PREFER_BUSY_POLL 69
#endif
#ifndef SO_BUSY_POLL_BUDGET
#define SO_BUSY_POLL_BUDGET 70
#endif

static int busy = 0;
static void opts(int fd) {
  int one = 1, bp = 50;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
  if (busy == 1 || busy == 2) setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &bp, sizeof bp);  // busy=3: spin recv only, no SO_BUSY_POLL
  if (busy == 4) { int big = 5000; setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &big, sizeof big); }  // busy=4: blocking recv, kernel busy-loops up to 5 ms
  if (busy == 2) { int bud = 64; if (setsockopt(fd, SOL_SOCKET, SO_PREFER_BUSY_POLL, &one, sizeof one)) perror("SO_PREFER_BUSY_POLL");
                   if (setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL_BUDGET, &bud, sizeof bud)) perror("SO_BUSY_POLL_BUDGET"); }
}
static int rd(int fd, char *b, size_t n) {
  size_t g = 0;
  while (g < n) {
    ssize_t r = recv(fd, b + g, n - g, (busy && busy != 4) ? MSG_DONTWAIT : 0);
    if (r > 0) { g += r; continue; }
    if (r == 0) return -1;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
    return -1;
  }
  return 0;
}
static int wr(int fd, const char *b, size_t n) {
  size_t g = 0;
  while (g < n) { ssize_t r = send(fd, b + g, n - g, 0); if (r < 0) { if (errno == EAGAIN || errno == EINTR) continue; return -1; } g += r; }
  return 0;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: see header\n"); return 2; }
  int port = atoi(argv[3]); size_t req = atol(argv[4]), rep = atol(argv[5]);
  struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, argv[2], &a.sin_addr);
  char *b = calloc(1, req > rep ? req : rep);
  if (argv[1][0] == 's') {
    busy = argc > 6 ? atoi(argv[6]) : 0; double dly = argc > 7 ? atof(argv[7]) * 1e-6 : 0;
    double hb = argc > 8 ? atof(argv[8]) * 1e-6 : 0;  // heartbeat: send 1 reply byte at once, then 1 byte every hb s while computing
    int l = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(l, (struct sockaddr *)&a, sizeof a) || listen(l, 1)) { perror("bind"); return 1; }
    int fd = accept(l, 0, 0); opts(fd);
    while (rd(fd, b, req) == 0) {
      size_t sent = 0; double t0 = now(), next = t0;
      if (hb > 0) { if (wr(fd, b, 1)) break; sent = 1; next = t0 + hb; }
      while (now() - t0 < dly) { if (hb > 0 && now() >= next && sent < rep - 1) { if (wr(fd, b, 1)) return 1; sent++; next += hb; } }
      if (wr(fd, b + sent, rep - sent)) break;
    }
    return 0;
  }
  int iters = atoi(argv[6]); busy = argc > 7 ? atoi(argv[7]) : 0;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (connect(fd, (struct sockaddr *)&a, sizeof a)) { perror("connect"); return 1; }
  opts(fd);
  double *t = malloc(sizeof(double) * iters);
  for (int i = 0; i < iters + 200; i++) {
    double t0 = now();
    if (wr(fd, b, req) || rd(fd, b, rep)) { fprintf(stderr, "io error\n"); return 1; }
    if (i >= 200) t[i - 200] = (now() - t0) * 1e6;
  }
  qsort(t, iters, sizeof(double), cmp);
  printf("req=%zu rep=%zu busy=%d p50=%.1fus p90=%.1fus p99=%.1fus\n", req, rep, busy, t[iters / 2], t[iters * 9 / 10], t[iters * 99 / 100]);
  return 0;
}
