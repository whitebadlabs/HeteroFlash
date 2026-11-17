// Batch-1..4 GEMV on the workstation CPU at the decode shapes of a dual-memory split (HeteroFlash, 2026-11-16).
//   gemv_cpu <fmt fp8|bf16|int8> <T> <iters> <layer_copies> <NxK> [NxK ...]
//   gemv_cpu check
// One "layer" = the listed matrices, run in order with a barrier after each (as dependent sub-phases would be).
// layer_copies distinct copies of the layer are cycled so weights come from DRAM, not the 128 MB L3.
// fp8: E4M3 weights, row-major [N][K], one E8M0 (power-of-two) scale per 32 weights along K, exactly the
//      checkpoint's format; decoded on the fly to bf16 with the scale folded into the exponent, then vdpbf16ps
//      against bf16 activations (exact for normal E4M3 values; E4M3 subnormals are flushed, a ~1/16 of codes
//      in [2^-9,2^-6] - affects values only, not bytes or timing). bf16: vdpbf16ps. int8: s8 weights x u8
//      activations, vpdpbusd, per-row scale (NOT the checkpoint format; a speed reference only).
// Threads: OMP_NUM_THREADS / OMP_PLACES / OMP_PROC_BIND; rows are split in blocks of 4 across threads.
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define AI static inline __attribute__((always_inline))
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
AI __m512i dec_fp8(__m256i b,int sc){
  const __m512i c80=_mm512_set1_epi16(0x80),c7f=_mm512_set1_epi16(0x7f);
  __m512i w=_mm512_cvtepu8_epi16(b);
  __m512i sgn=_mm512_slli_epi16(_mm512_and_si512(w,c80),8);
  __m512i em=_mm512_slli_epi16(_mm512_and_si512(w,c7f),4);
  __mmask32 nz=_mm512_cmpgt_epu16_mask(em,_mm512_set1_epi16(0x7f)); // exp field != 0 (normal); 0 and subnormal -> 0
  __m512i off=_mm512_set1_epi16((short)((sc-127+120)<<7));
  return _mm512_or_si512(_mm512_maskz_add_epi16(nz,em,off),sgn);
}
AI void k_fp8(const uint8_t*W,const uint8_t*S,int N,int K,const uint16_t*x,const int T,float*y,int r0,int r1){
  int KB=K/32;
  for(int r=r0;r<r1;r+=4){ __m512 a[4][4]; for(int i=0;i<4;i++)for(int t=0;t<4;t++)a[i][t]=_mm512_setzero_ps();
    const uint8_t*w0=W+(size_t)r*K; const uint8_t*s0=S+(size_t)r*KB;
    for(int k=0;k<K;k+=32){ __m512i xv[4]; for(int t=0;t<T;t++) xv[t]=_mm512_loadu_si512(x+(size_t)t*K+k);
      for(int i=0;i<4;i++){ __m512i bf=dec_fp8(_mm256_loadu_si256((const void*)(w0+(size_t)i*K+k)),s0[(size_t)i*KB+k/32]);
        for(int t=0;t<T;t++) a[i][t]=_mm512_dpbf16_ps(a[i][t],(__m512bh)bf,(__m512bh)xv[t]); } }
    for(int i=0;i<4;i++)for(int t=0;t<T;t++) y[(size_t)t*N+r+i]=_mm512_reduce_add_ps(a[i][t]); }
}
AI void k_bf16(const uint16_t*W,int N,int K,const uint16_t*x,const int T,float*y,int r0,int r1){
  for(int r=r0;r<r1;r+=4){ __m512 a[4][4]; for(int i=0;i<4;i++)for(int t=0;t<4;t++)a[i][t]=_mm512_setzero_ps();
    const uint16_t*w0=W+(size_t)r*K;
    for(int k=0;k<K;k+=32){ __m512i xv[4]; for(int t=0;t<T;t++) xv[t]=_mm512_loadu_si512(x+(size_t)t*K+k);
      for(int i=0;i<4;i++){ __m512i w=_mm512_loadu_si512(w0+(size_t)i*K+k);
        for(int t=0;t<T;t++) a[i][t]=_mm512_dpbf16_ps(a[i][t],(__m512bh)w,(__m512bh)xv[t]); } }
    for(int i=0;i<4;i++)for(int t=0;t<T;t++) y[(size_t)t*N+r+i]=_mm512_reduce_add_ps(a[i][t]); }
}
AI void k_int8(const int8_t*W,const float*rs,int N,int K,const uint8_t*x,const int T,float*y,int r0,int r1){
  for(int r=r0;r<r1;r+=4){ __m512i a[4][4]; for(int i=0;i<4;i++)for(int t=0;t<4;t++)a[i][t]=_mm512_setzero_si512();
    const int8_t*w0=W+(size_t)r*K;
    for(int k=0;k<K;k+=64){ __m512i xv[4]; for(int t=0;t<T;t++) xv[t]=_mm512_loadu_si512(x+(size_t)t*K+k);
      for(int i=0;i<4;i++){ __m512i w=_mm512_loadu_si512(w0+(size_t)i*K+k);
        for(int t=0;t<T;t++) a[i][t]=_mm512_dpbusd_epi32(a[i][t],xv[t],w); } }
    for(int i=0;i<4;i++)for(int t=0;t<T;t++) y[(size_t)t*N+r+i]=rs[r+i]*(float)_mm512_reduce_add_epi32(a[i][t]); }
}
typedef struct{int N,K;void*W;uint8_t*S;float*rs;}mat;
static int fmt; // 0 fp8 1 bf16 2 int8
static void run(mat*m,const void*x,int T,float*y,int r0,int r1){
  #define DISPATCH(TT) \
    if(fmt==0) k_fp8(m->W,m->S,m->N,m->K,x,TT,y,r0,r1); else if(fmt==1) k_bf16(m->W,m->N,m->K,x,TT,y,r0,r1); else k_int8(m->W,m->rs,m->N,m->K,x,TT,y,r0,r1);
  switch(T){case 1:DISPATCH(1);break;case 2:DISPATCH(2);break;case 3:DISPATCH(3);break;default:DISPATCH(4);}
}
static size_t mbytes(mat*m){size_t e=(size_t)m->N*m->K; return fmt==0? e+e/32 : fmt==1? 2*e : e+4*m->N;}
static uint16_t f2bf(float f){uint32_t u;memcpy(&u,&f,4);return u>>16;}
static float bf2f(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static float e4m3(uint8_t b){int s=b>>7,e=(b>>3)&15,m=b&7; if(e==0) return 0.f; float v=ldexpf(1.f+m/8.f,e-7); return s?-v:v;}
#include <math.h>
static void alloc(mat*m,int N,int K,unsigned seed){
  m->N=N;m->K=K; size_t e=(size_t)N*K; m->W=aligned_alloc(64,(fmt==1?2*e:e)+64); m->S=aligned_alloc(64,e/32+64); m->rs=aligned_alloc(64,4*N+64);
  #pragma omp parallel
  { unsigned s=seed*7919u+omp_get_thread_num()*104729u+1;
    #pragma omp for schedule(static)
    for(size_t i=0;i<e;i++){ s=s*1103515245u+12345u; uint8_t b=(s>>16)&0xff; if(((b>>3)&15)==15&&(b&7)==7) b^=1; // avoid NaN
      if(fmt==1) ((uint16_t*)m->W)[i]=f2bf(e4m3(b)); else ((uint8_t*)m->W)[i]=b; }
    #pragma omp for schedule(static)
    for(size_t i=0;i<e/32;i++) m->S[i]=120+(i%8);
  }
  for(int i=0;i<N;i++) m->rs[i]=1e-3f;
}
static int check(void){
  int N=64,K=512,T=4,bad=0; for(int f=0;f<2;f++){ fmt=f; mat m; alloc(&m,N,K,3);
    uint16_t*x=aligned_alloc(64,2*T*K); float*xf=malloc(4*T*K); for(int i=0;i<T*K;i++){xf[i]=bf2f(f2bf(sinf(i*0.37f)));x[i]=f2bf(xf[i]);}
    float*y=calloc(T*N,4); run(&m,x,T,y,0,N); double num=0,den=0;
    for(int t=0;t<T;t++)for(int n=0;n<N;n++){ double r=0; for(int k=0;k<K;k++){ float w= f==0? e4m3(((uint8_t*)m.W)[(size_t)n*K+k])*ldexpf(1.f,m.S[(size_t)n*(K/32)+k/32]-127) : bf2f(((uint16_t*)m.W)[(size_t)n*K+k]); r+=(double)w*xf[t*K+k]; }
      num+=(y[t*N+n]-r)*(y[t*N+n]-r); den+=r*r; }
    printf("check %s: rel L2 error %.2e\n",f?"bf16":"fp8",sqrt(num/den)); if(sqrt(num/den)>1e-5) bad=1; }
  return bad;
}
int main(int c,char**v){
  if(c>1&&!strcmp(v[1],"check")) return check();
  if(c<6){fprintf(stderr,"usage: see header\n");return 2;}
  fmt=!strcmp(v[1],"fp8")?0:!strcmp(v[1],"bf16")?1:2; int T=atoi(v[2]),iters=atoi(v[3]),L=atoi(v[4]),nm=c-5;
  mat*ms=malloc(sizeof(mat)*L*nm); size_t lb=0; int maxN=0,maxK=0;
  for(int j=0;j<nm;j++){int N,K; sscanf(v[5+j],"%dx%d",&N,&K); if(N>maxN)maxN=N; if(K>maxK)maxK=K; for(int l=0;l<L;l++) alloc(&ms[l*nm+j],N,K,l*31+j);}
  for(int j=0;j<nm;j++) lb+=mbytes(&ms[j]);
  uint16_t*x=aligned_alloc(64,2*(size_t)4*maxK); for(int i=0;i<4*maxK;i++) x[i]=f2bf(0.01f*(i%97));
  float*y=aligned_alloc(64,4*(size_t)4*maxN);
  double*lt=malloc(sizeof(double)*iters); int nt=omp_get_max_threads();
  #pragma omp parallel
  { int tid=omp_get_thread_num();
    for(int it=-L;it<iters;it++){ int l=((it%L)+L)%L; double t0=0;
      #pragma omp barrier
      #pragma omp master
      t0=now();
      for(int j=0;j<nm;j++){ mat*m=&ms[l*nm+j]; int blk=((m->N/4+nt-1)/nt)*4; int r0=tid*blk, r1=r0+blk; if(r1>m->N) r1=m->N;
        if(r0<r1) run(m,x,T,y,r0,r1);
        #pragma omp barrier
      }
      #pragma omp master
      if(it>=0) lt[it]=now()-t0;
    } }
  double s=0; for(int i=0;i<iters;i++) s+=lt[i];
  int cmpd(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);} qsort(lt,iters,sizeof(double),cmpd);
  printf("fmt=%s T=%d threads=%d layer_MB=%.2f pool_GB=%.2f mats=",v[1],T,nt,lb/1e6,lb*(double)L/1e9);
  for(int j=0;j<nm;j++) printf("%s%s",v[5+j],j<nm-1?",":"");
  printf(" p50_us=%.1f mean_us=%.1f p90_us=%.1f GBps_p50=%.1f\n",lt[iters/2]*1e6,s/iters*1e6,lt[iters*9/10]*1e6,lb/lt[iters/2]/1e9);
  return 0;
}
