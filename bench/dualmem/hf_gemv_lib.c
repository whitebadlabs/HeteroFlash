// Shared library for the head-split exactness check: the same CPU GEMV kernels as gemv_cpu.c, callable from Python.
//   hf_gemv(fmt, W, S8, SF, N, K, x_bf16[T][K], T, y_f32[T][N])
//   fmt 0: FP8 E4M3 + E8M0 scale per 32 along K (MXFP8, S8 [N][K/32]); 1: BF16; 3: FP8 E4M3 + FP32 128x128 block
//   scales (SF [N/128][K/128], DeepSeek-style; dot over each 128-column block, then scaled and accumulated in fp32).
// Threads from OMP_NUM_THREADS; rows split across threads in blocks of 4.
#define main gemv_cpu_main
#include "gemv_cpu.c"
#undef main
AI void k_fp8b(const uint8_t*W,const float*SF,int N,int K,const uint16_t*x,const int T,float*y,int r0,int r1){
  int KB=K/128;
  for(int r=r0;r<r1;r+=4){ __m512 a[4][4]; for(int i=0;i<4;i++)for(int t=0;t<4;t++)a[i][t]=_mm512_setzero_ps();
    const uint8_t*w0=W+(size_t)r*K; const float*sr=SF+(size_t)(r/128)*KB;
    for(int kb=0;kb<K;kb+=128){ __m512 tm[4][4]; for(int i=0;i<4;i++)for(int t=0;t<4;t++)tm[i][t]=_mm512_setzero_ps();
      for(int k=kb;k<kb+128;k+=32){ __m512i xv[4]; for(int t=0;t<T;t++) xv[t]=_mm512_loadu_si512(x+(size_t)t*K+k);
        for(int i=0;i<4;i++){ __m512i bf=dec_fp8(_mm256_loadu_si256((const void*)(w0+(size_t)i*K+k)),127);
          for(int t=0;t<T;t++) tm[i][t]=_mm512_dpbf16_ps(tm[i][t],(__m512bh)bf,(__m512bh)xv[t]); } }
      __m512 s=_mm512_set1_ps(sr[kb/128]);
      for(int i=0;i<4;i++)for(int t=0;t<T;t++) a[i][t]=_mm512_fmadd_ps(tm[i][t],s,a[i][t]); }
    for(int i=0;i<4;i++)for(int t=0;t<T;t++) y[(size_t)t*N+r+i]=_mm512_reduce_add_ps(a[i][t]); }
}
static void run_b(const uint8_t*W,const float*SF,int N,int K,const uint16_t*x,int T,float*y,int r0,int r1){
  switch(T){case 1:k_fp8b(W,SF,N,K,x,1,y,r0,r1);break;case 2:k_fp8b(W,SF,N,K,x,2,y,r0,r1);break;case 3:k_fp8b(W,SF,N,K,x,3,y,r0,r1);break;default:k_fp8b(W,SF,N,K,x,4,y,r0,r1);}
}
int hf_gemv(int f,const void*W,const uint8_t*S8,const float*SF,int N,int K,const uint16_t*x,int T,float*y){
  if(N%4||K%(f==3?128:32)) return -1;
  float*tmp=aligned_alloc(64,sizeof(float)*4*(size_t)N+64);
  for(int t0=0;t0<T;t0+=4){ int tt=T-t0<4?T-t0:4; const uint16_t*xx=x+(size_t)t0*K;
    #pragma omp parallel
    { int nt=omp_get_num_threads(),tid=omp_get_thread_num(); int blk=((N/4+nt-1)/nt)*4; int r0=tid*blk,r1=r0+blk; if(r1>N)r1=N;
      if(r0<r1){ if(f==3) run_b(W,SF,N,K,xx,tt,tmp,r0,r1);
        else { mat m={N,K,(void*)W,(uint8_t*)S8,0}; fmt=f; run(&m,xx,tt,tmp,r0,r1);} } }
    for(int t=0;t<tt;t++) memcpy(y+(size_t)(t0+t)*N,tmp+(size_t)t*N,sizeof(float)*N); }
  free(tmp); return 0;
}
