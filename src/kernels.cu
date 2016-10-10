/* kernels.cu

   Custom ACMI CUDA kernels. */

#include "acmi.h"

namespace {

enum { SUM_SWEEP_FACTOR = 4 };

/* Kernel parameters. */
int g_maxBlocksPerGrid, g_maxThreadsPerBlock,
    g_blocksPerKernel,  g_threadsPerBlock,
    g_threadsPerKernel;

/* Copies elements from one nxn matrix to another, converting them to the
   precision of the destination matrix. */
template<typename T, typename U> __global__ void
kernCopy(T* dst, const U* src, const int64_t n2) {
  const int offset = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  const T* end = dst + n2;
  src += offset;
  dst += offset;

  for (; dst < end; dst += stride, src += stride) {
    *dst = *src;
  }
}

template<typename T> __global__ void
kernAddId(T* a, const T alpha, const int n) {
  const T* end = a + n*n;
  a += (blockIdx.x*blockDim.x + threadIdx.x)*(n + 1);
  const int stride = gridDim.x*blockDim.x*(n + 1);

  for (; a < end; a += stride) {
    *a += alpha;
  }
}

} // end anonymous namespace

extern "C" {

/* Sets up kernel parameters. */
void cuSetUp(const int maxBlocksPerKernel, const int n) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // assumes usage of the first device
  debug("setting up kernels on %s", prop.name);
  g_maxBlocksPerGrid = prop.maxGridSize[0];
  g_maxThreadsPerBlock = prop.maxThreadsPerBlock;
  const int64_t n2 = n*n;
  g_blocksPerKernel = (n2 + g_maxThreadsPerBlock - 1) / g_maxThreadsPerBlock;
  g_blocksPerKernel = iMin(maxBlocksPerKernel, g_blocksPerKernel);
  g_threadsPerBlock = iMin(g_maxThreadsPerBlock, n2);
  g_threadsPerKernel = g_threadsPerBlock*g_blocksPerKernel;
  debug("max  blocks/grid  : %d\n"
        "max threads/block : %d\n"
        "     blocks/kernel: %d\n"
        "    threads/block : %d\n"
        "    threads/kernel: %d", g_maxBlocksPerGrid, g_maxThreadsPerBlock,
        g_blocksPerKernel, g_threadsPerBlock, g_threadsPerKernel);
}

void cuShutDown() {
  debug("shutting down kernels");
}

/* Doubles matrix precision. */
void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  kernCopy<<<g_blocksPerKernel, g_threadsPerBlock>>>
    ((double*)dst, (const float*)src, n2);
}

/* Adds alpha*I to the nxn matrix backed by the device array elems. */
void cuAddId(void* elems, double alpha, int n, int elemSize) {
  const int nThreads = iMin(g_maxThreadsPerBlock, n);
  switch (elemSize) {
  case 4: kernAddId<<<1, nThreads>>>((float*)elems, (float)alpha, n); break;
  case 8: kernAddId<<<1, nThreads>>>((double*)elems, alpha, n);       break;
  }
}

} // end extern "C"
