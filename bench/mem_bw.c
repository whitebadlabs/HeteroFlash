#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>
int main(){ size_t n=(size_t)8<<30; n/=8; /* 8 GB */
 uint64_t *a=malloc(n*8);
 #pragma omp parallel for
 for(size_t i=0;i<n;i++) a[i]=i;
 int ts[]={8,16,32,64};
 for(int k=0;k<4;k++){ omp_set_num_threads(ts[k]); double best=0;
  for(int r=0;r<4;r++){ uint64_t s=0; double t=omp_get_wtime();
   #pragma omp parallel for reduction(+:s)
   for(size_t i=0;i<n;i++) s+=a[i];
   t=omp_get_wtime()-t; double bw=n*8/t/1e9; if(bw>best)best=bw; if(s==42)puts("");}
  printf("%d threads: read %.0f GB/s\n",ts[k],best);}
 return 0;}
