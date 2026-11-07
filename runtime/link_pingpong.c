// Direct-link round-trip microbenchmark: request of REQ bytes, reply of REP bytes, TCP_NODELAY, optional busy polling.
//   server: link_pingpong s <bind_ip> <port> <req> <rep> [busy]
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

static int busy = 0;
static void opts(int fd) {
  int one = 1, bp = 50;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
  if (busy) setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &bp, sizeof bp);
}
static int rd(int fd, char *b, size_t n) {
  size_t g = 0;
  while (g < n) {
    ssize_t r = recv(fd, b + g, n - g, busy ? MSG_DONTWAIT : 0);
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
    busy = argc > 6 ? atoi(argv[6]) : 0;
    int l = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(l, (struct sockaddr *)&a, sizeof a) || listen(l, 1)) { perror("bind"); return 1; }
    int fd = accept(l, 0, 0); opts(fd);
    while (rd(fd, b, req) == 0) if (wr(fd, b, rep)) break;
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
