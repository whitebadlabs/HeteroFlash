// Symmetric per-phase partial-sum swap over TCP, the exchange pattern of the Bet 1 dual-memory split.
// Each phase: both sides "compute" (spin compute_us), then each sends its MSG-byte partial and receives the other's.
// A decode cycle = PHASES phases (2 per layer x 45 layers = 90). The client reports cycle time and overhead.
//   server: phase_swap s <bind_ip> <port> <msg> <compute_us> <mode> [hb_us]
//   client: phase_swap c <ip> <port> <msg> <compute_us> <mode> <phases> <cycles> [hb_us]
// mode 0: blocking recv;  mode 1: SO_BUSY_POLL 50 us + non-blocking spin recv.
// hb_us > 0: while computing, send a 0x00 heartbeat byte every hb_us (receiver skips 0x00 before the 0xAA frame header).
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
static int mode=0; static double hb=0;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void opts(int fd){int one=1,bp=50,buf=4<<20; setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one); setsockopt(fd,IPPROTO_TCP,TCP_QUICKACK,&one,sizeof one);
  setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&buf,sizeof buf); setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&buf,sizeof buf);
  if(mode==1) setsockopt(fd,SOL_SOCKET,SO_BUSY_POLL,&bp,sizeof bp);}
static int rdn(int fd,char*b,size_t n){size_t g=0; while(g<n){ssize_t r=recv(fd,b+g,n-g,mode?MSG_DONTWAIT:0); if(r>0){g+=r;continue;} if(r==0)return -1;
  if(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR)continue; return -1;} if(mode==0){int one=1;setsockopt(fd,IPPROTO_TCP,TCP_QUICKACK,&one,sizeof one);} return 0;}
static int wrn(int fd,const char*b,size_t n){size_t g=0; while(g<n){ssize_t r=send(fd,b+g,n-g,0); if(r<0){if(errno==EAGAIN||errno==EINTR)continue;return -1;} g+=r;} return 0;}
static int rdmsg(int fd,char*b,size_t n){char h; do{ if(rdn(fd,&h,1))return -1; }while(h==0); if(h!=(char)0xAA){fprintf(stderr,"bad frame %d\n",h);return -1;} return rdn(fd,b,n);}
static int compute(int fd,double us){double t0=now(),nx=t0+hb*1e-6; char z=0; while((now()-t0)*1e6<us){ if(hb>0&&now()>=nx){ if(wrn(fd,&z,1))return -1; nx+=hb*1e-6; } } return 0;}
static int phase(int fd,char*s,char*r,size_t m,double us){ if(compute(fd,us))return -1; s[0]=(char)0xAA; if(wrn(fd,s,m+1))return -1; return rdmsg(fd,r,m);}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int c,char**v){
  if(c<7){fprintf(stderr,"usage: see header\n");return 2;}
  int port=atoi(v[3]); size_t m=atol(v[4]); double cu=atof(v[5]); mode=atoi(v[6]);
  struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,v[2],&a.sin_addr);
  char*s=calloc(1,m+1),*r=calloc(1,m+1); for(size_t i=1;i<=m;i++)s[i]=(char)(i*7+1);
  if(v[1][0]=='s'){ hb=c>7?atof(v[7]):0; int l=socket(AF_INET,SOCK_STREAM,0),one=1; setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    if(bind(l,(struct sockaddr*)&a,sizeof a)||listen(l,1)){perror("bind");return 1;} int fd=accept(l,0,0); opts(fd);
    // server: wait for the client's start frame, then run the same symmetric phase loop until EOF
    if(rdmsg(fd,r,m))return 1; while(1){ if(phase(fd,s,r,m,cu))break; } return 0; }
  int P=atoi(v[7]),C=atoi(v[8]); hb=c>9?atof(v[9]):0;
  int fd=socket(AF_INET,SOCK_STREAM,0); if(connect(fd,(struct sockaddr*)&a,sizeof a)){perror("connect");return 1;} opts(fd);
  // Both sides run: compute (concurrently), send own partial, receive the other's. Steady-state phase time is
  // compute + one-way delivery (incl. serialisation) + skew, so overhead = phase - compute.
  s[0]=(char)0xAA; if(wrn(fd,s,m+1))return 1;
  double*ct=malloc(sizeof(double)*C), *pt=malloc(sizeof(double)*C*P); int np=0;
  for(int it=-3;it<C;it++){ double t0=now();
    for(int p=0;p<P;p++){ double p0=now(); if(phase(fd,s,r,m,cu)){fprintf(stderr,"io\n");return 1;} if(it>=0) pt[np++]=(now()-p0)*1e6; }
    if(it>=0) ct[it]=(now()-t0)*1e3; }
  qsort(ct,C,sizeof(double),cmp); qsort(pt,np,sizeof(double),cmp);
  printf("msg=%zu compute_us=%.0f mode=%d hb_us=%.0f phases=%d cycle_ms_p50=%.3f p90=%.3f overhead_per_phase_us_p50=%.1f phase_us_p50=%.1f p99=%.1f\n",
    m,cu,mode,hb,P,ct[C/2],ct[C*9/10],(ct[C/2]*1e3-P*cu)/P,pt[np/2],pt[np*99/100]);
  return 0;
}
