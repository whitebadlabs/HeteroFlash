// Simulate cold-expert MoE FFN on CPU: E experts, each gate/up [2048x4096] + down [4096x2048], int8 weights (VNNI) or int4 (nibbles).
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#define H 4096
#define I 2048
#define NEXP 96   /* distinct experts in memory per "layer" pool, cycle to defeat cache */
static inline int dot_s8(const uint8_t*x,const int8_t*w,int n){__m512i acc=_mm512_setzero_si512();
 for(int k=0;k<n;k+=64) acc=_mm512_dpbusd_epi32(acc,_mm512_loadu_si512(x+k),_mm512_loadu_si512(w+k));
 return _mm512_reduce_add_epi32(acc);}
static inline int dot_s4(const uint8_t*x,const uint8_t*w,int n){__m512i acc=_mm512_setzero_si512();const __m512i m=_mm512_set1_epi8(0x0f),o=_mm512_set1_epi8(8);
 for(int k=0;k<n;k+=128){__m512i p=_mm512_loadu_si512(w+k/2);
  __m512i lo=_mm512_sub_epi8(_mm512_and_si512(p,m),o), hi=_mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(p,4),m),o);
  acc=_mm512_dpbusd_epi32(acc,_mm512_loadu_si512(x+k),lo); acc=_mm512_dpbusd_epi32(acc,_mm512_loadu_si512(x+k+64),hi);}
 return _mm512_reduce_add_epi32(acc);}
int main(int argc,char**argv){int bits=atoi(argv[1]); int E=atoi(argv[2]); int T=atoi(argv[3]); /* experts per token-layer, tokens */
 size_t per=(size_t)3*H*I*bits/8; uint8_t*W=aligned_alloc(64,per*NEXP); memset(W,0x11,per*NEXP);
 uint8_t*x=aligned_alloc(64,H*T*2+8192); memset(x,3,H*T*2+8192); int32_t*y=aligned_alloc(64,H*T*4*E+65536);
 int iters=200; int distinct=E*T<NEXP?E*T:NEXP; /* union of experts touched (worst case: all different) */
 double t0=0; for(int it=-20;it<iters;it++){ if(it==0)t0=omp_get_wtime(); int base=((it+1000)*distinct)%NEXP;
  // gate+up: rows 2*I over H, for each distinct expert, each of its tokens (approx: each expert used by T*E/distinct tokens)
  int tok_per=(E*T+distinct-1)/distinct;
  #pragma omp parallel for collapse(2) schedule(static)
  for(int e=0;e<distinct;e++) for(int r=0;r<2*I;r+=16){ uint8_t*we=W+per*((base+e)%NEXP);
    for(int rr=r;rr<r+16;rr++) for(int t=0;t<tok_per;t++){ int v= bits==8? dot_s8(x+t*H,(int8_t*)we+(size_t)rr*H,H):dot_s4(x+t*H,we+(size_t)rr*H/2,H); y[(e*tok_per+t)%(H*T)]+=v;}}
  #pragma omp parallel for collapse(2) schedule(static)
  for(int e=0;e<distinct;e++) for(int r=0;r<H;r+=16){ uint8_t*we=W+per*((base+e)%NEXP)+(size_t)2*I*H*bits/8;
    for(int rr=r;rr<r+16;rr++) for(int t=0;t<tok_per;t++){ int v= bits==8? dot_s8(x+t*I,(int8_t*)we+(size_t)rr*I,I):dot_s4(x+t*I,we+(size_t)rr*I/2,I); y[(e*tok_per+t)%(H*T)]+=v;}}
 }
 double dt=(omp_get_wtime()-t0)/iters; double gb=(double)per*distinct/1e9;
 printf("int%d experts/tok=%d tokens=%d distinct=%d threads=%d: %.3f ms/layer, %.0f GB/s weights\n",bits,E,T,distinct,omp_get_max_threads(),dt*1e3,gb/dt);
 return y[5]==12345;}
