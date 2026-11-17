// Bet 1 one-layer-split prototype, workstation side: per phase, compute this side's FP8 slice (real E4M3+E8M0
// GEMV kernels from gemv_cpu.c), then swap a T x 4096 bf16 partial with the Spark over TCP (busy-poll spin recv).
//   dual_srv <bind_ip> <port> <T> <layer_copies> <phaseA NxK,NxK,...> <phaseB NxK,...>
// Phases alternate A,B,A,B,... cycling through layer_copies distinct weight copies (DRAM-resident, not L3).
// Prints per-phase compute time stats on exit (client closes the connection).
#define main gemv_main
#include "gemv_cpu.c"
#undef main
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
static int rdn(int fd,char*b,size_t n){size_t g=0; while(g<n){ssize_t r=recv(fd,b+g,n-g,MSG_DONTWAIT); if(r>0){g+=r;continue;} if(r==0)return -1;
  if(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR)continue; return -1;} return 0;}
static int wrn(int fd,const char*b,size_t n){size_t g=0; while(g<n){ssize_t r=send(fd,b+g,n-g,0); if(r<0){if(errno==EAGAIN||errno==EINTR)continue;return -1;} g+=r;} return 0;}
static int parse(const char*s,int*N,int*K){int n=0; const char*p=s; while(*p){ if(sscanf(p,"%dx%d",&N[n],&K[n])==2) n++; while(*p&&*p!=',')p++; if(*p)p++; } return n;}
int main(int c,char**v){
  if(c<7){fprintf(stderr,"usage: see header\n");return 2;}
  fmt=0; int T=atoi(v[3]),L=atoi(v[4]); int NA[64],KA[64],NB[64],KB[64]; int na=parse(v[5],NA,KA),nb=parse(v[6],NB,KB);
  mat*A=malloc(sizeof(mat)*L*na),*B=malloc(sizeof(mat)*L*nb); int maxN=0,maxK=0;
  for(int l=0;l<L;l++){ for(int j=0;j<na;j++){alloc(&A[l*na+j],NA[j],KA[j],l*13+j); if(NA[j]>maxN)maxN=NA[j]; if(KA[j]>maxK)maxK=KA[j];}
                         for(int j=0;j<nb;j++){alloc(&B[l*nb+j],NB[j],KB[j],l*17+j+5); if(NB[j]>maxN)maxN=NB[j]; if(KB[j]>maxK)maxK=KB[j];} }
  size_t ab=0,bb=0; for(int j=0;j<na;j++)ab+=mbytes(&A[j]); for(int j=0;j<nb;j++)bb+=mbytes(&B[j]);
  uint16_t*x=aligned_alloc(64,2*(size_t)4*maxK); for(int i=0;i<4*maxK;i++) x[i]=f2bf(0.01f*(i%97));
  float*y=aligned_alloc(64,4*(size_t)4*maxN); size_t m=(size_t)T*4096*2; char*sb=calloc(1,m+1),*rb=calloc(1,m+1);
  struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(atoi(v[2])); inet_pton(AF_INET,v[1],&a.sin_addr);
  int ls=socket(AF_INET,SOCK_STREAM,0),one=1,bp=50,buf=4<<20; setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
  if(bind(ls,(struct sockaddr*)&a,sizeof a)||listen(ls,1)){perror("bind");return 1;}
  fprintf(stderr,"dual_srv ready: phaseA %.1f MB, phaseB %.1f MB, %d copies, threads %d\n",ab/1e6,bb/1e6,L,omp_get_max_threads());
  int fd=accept(ls,0,0); setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one); setsockopt(fd,SOL_SOCKET,SO_BUSY_POLL,&bp,sizeof bp);
  setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&buf,sizeof buf); setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&buf,sizeof buf);
  int nt=omp_get_max_threads(); long ph=0; double*ct=malloc(sizeof(double)*2000000); char h;
  // wait for the start frame (header byte + m bytes), then: compute, send, receive, ...
  if(rdn(fd,&h,1)||rdn(fd,rb,m)) return 1;
  while(1){ int isA=(ph%2==0); int l=(ph/2)%L; mat*ms=isA?&A[l*na]:&B[l*nb]; int n=isA?na:nb;
    double t0=now();
    #pragma omp parallel
    { int tid=omp_get_thread_num();
      for(int j=0;j<n;j++){ mat*mm=&ms[j]; int blk=((mm->N/4+nt-1)/nt)*4; int r0=tid*blk,r1=r0+blk; if(r1>mm->N)r1=mm->N; if(r0<r1) run(mm,x,T,y,r0,r1);
        #pragma omp barrier
      } }
    if(ph<2000000) ct[ph]=now()-t0;
    for(int t=0;t<T;t++) for(int i=0;i<4096;i++) ((uint16_t*)(sb+1))[t*4096+i]=f2bf(y[(size_t)t*4096+i]);
    sb[0]=(char)0xAA; if(wrn(fd,sb,m+1)) break; if(rdn(fd,&h,1)||rdn(fd,rb,m)) break; ph++; }
  double sa=0,sbb=0; long na_=0,nb_=0; for(long i=0;i<ph&&i<2000000;i++){ if(i%2==0){sa+=ct[i];na_++;} else {sbb+=ct[i];nb_++;} }
  fprintf(stderr,"dual_srv phases=%ld cpu_compute_mean_us A=%.1f (%.0f GB/s) B=%.1f (%.0f GB/s)\n",ph,sa/na_*1e6,ab/(sa/na_)/1e9,sbb/nb_*1e6,bb/(sbb/nb_)/1e9);
  return 0;
}
