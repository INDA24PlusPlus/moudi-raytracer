#include "rand.h"
#include <math.h>
#include <stdio.h>

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

// A and C are constants
__device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
  return z = (A * z + C);
}

__device__ unsigned z1, z2, z3, z4;
__device__ double HybridTaus()
{
  // Combined period is lcm(p1,p2,p3,p4) ~= 2^121
   return 2.3283064365387e-10 * (              // Periods
    TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
    TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
    TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
    LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32
   );
}

__device__ float rand_float()
{
    return HybridTaus();
    // double r = sqrt(-2.0 * log(HybridTaus()));
    // double theta = 2.0 * M_PI * HybridTaus();
    // return r * cos(theta);
}
