// Timestamped CPU read-bandwidth stream for overlap tests (HeteroFlash dual-memory, 2026-11-16).
//   stream_cpu <buf_GB> <start_epoch> <duration_s> <window_s> <tag>
// Threads/pinning from OMP_NUM_THREADS / OMP_PLACES / OMP_PROC_BIND. Waits until start_epoch, then reads the buffer
// repeatedly (uint64 reduce) and prints "<tag> <epoch_mid> <GB/s>" per window. Each thread reads its own static chunk.
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
static double ep(void){struct timespec t;clock_gettime(CLOCK_REALTIME,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int c,char**v){ if(c<6){fprintf(stderr,"usage\n");return 2;}
 size_t n=(size_t)(atof(v[1])*(1<<30))/8; double st=atof(v[2]),dur=atof(v[3]),win=atof(v[4]); const char*tag=v[5];
 uint64_t*a=aligned_alloc(4096,n*8);
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;i++)a[i]=i;
 int nt=omp_get_max_threads(); fprintf(stderr,"%s threads=%d buf=%.1fGB ready, start in %.1fs\n",tag,nt,n*8/1e9,st-ep());
 while(ep()<st) usleep(1000);
 volatile uint64_t sink=0; double end=st+dur;
 while(ep()<end){ double t0=ep(); size_t bytes=0;
  while(ep()-t0<win){ uint64_t s=0;
   #pragma omp parallel for schedule(static) reduction(+:s)
   for(size_t i=0;i<n;i++) s+=a[i];
   sink+=s; bytes+=n*8; }
  double t1=ep(); printf("%s %.3f %.1f\n",tag,(t0+t1)/2,bytes/(t1-t0)/1e9); fflush(stdout); }
 return (int)(sink==1);
}
