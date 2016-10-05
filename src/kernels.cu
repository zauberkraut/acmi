/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <stdint.h>
#include "acmi.h"

namespace {

int g_maxBlocksPerGrid, g_maxThreadsPerBlock,
    g_blocksPerKernel,  g_threadsPerBlock,
    g_threadsPerKernel;

template<typename T, typename U> __global__ void
kernCopy(T* dst, const U* src, const int64_t n2) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const T* end = dst + n2;
  src += offset;
  dst += offset;

  for (; dst < end; dst += stride, src += stride) {
    *dst = *src;
  }
}

template<typename T> __global__ void
kernAddId(T* a, const T alpha, const int n) {
  a += (blockIdx.x * blockDim.x + threadIdx.x) * (n + 1);
  const int stride = gridDim.x * blockDim.x * (n + 1);
  const T* end = a + n * n;

  for (; a < end; a += stride) {
    *a += alpha;
  }
}

template<typename T> __global__ void
kernSweepSquares(const T* const a, double* const buckets, const bool subFromI,
                 const int n) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int64_t n2 = n * n;

  double sum = 0;

  for (int64_t i = offset; i < n2; i += stride) {
    int diag = subFromI && i % n == i / n;
    double e = diag - a[i];
    sum += e*e;
  }

  buckets[offset] = sum;
}

template<typename T> __global__ void
kernSweepSums(T* const buckets, const int bucketsLen) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double sumsum = 0;
  for (int i = offset; i < bucketsLen; i += stride) {
    sumsum += buckets[i];
  }

  buckets[offset] = sumsum;
}

} // end anonymous namespace

extern "C" {

void cuSetUp(const int maxBlocksPerKernel, const int n) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // assumes usage of the first device
  debug("setting up kernels on %s", prop.name);
  g_maxBlocksPerGrid = prop.maxGridSize[0];
  g_maxThreadsPerBlock = prop.maxThreadsPerBlock;
  const int64_t n2 = n * n;
  g_blocksPerKernel = (n2 + g_maxThreadsPerBlock - 1) / g_maxThreadsPerBlock;
  g_blocksPerKernel = iMin(maxBlocksPerKernel, g_blocksPerKernel);
  g_threadsPerBlock = iMin(g_maxThreadsPerBlock, n2);
  g_threadsPerKernel = g_threadsPerBlock * g_blocksPerKernel;
  debug("  max  blocks/grid  : %d\n"
        "  max threads/block : %d\n"
        "       blocks/kernel: %d\n"
        "      threads/block : %d\n"
        "      threads/kernel: %d", g_maxBlocksPerGrid, g_maxThreadsPerBlock,
        g_blocksPerKernel, g_threadsPerBlock, g_threadsPerKernel);
}

void cuShutDown() {
  debug("shutting down kernels");
}

void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  switch (srcElemSize) {
  case 4:
    kernCopy<<<g_blocksPerKernel, g_threadsPerBlock>>>
      ((double*)dst, (const float*)src, n2);
    break;
  case 8: /* WIP */; break;
  }
  CUCHECK;
}

void cuAddId(void* elems, double alpha, int n, int elemSize) {
  const int nThreads = iMin(g_maxThreadsPerBlock, n);
  switch (elemSize) {
  case 4: kernAddId<<<1, nThreads>>>((float*)elems, (float)alpha, n); break;
  case 8: kernAddId<<<1, nThreads>>>((double*)elems, alpha, n);       break;
  }
  CUCHECK;
}

// TODO: test manual gemm
// TODO: test combined kernels
double cuFroNorm(void* elems, bool subFromI, int n, int elemSize) {
  static double* buckets =
    (double*)cuMalloc(sizeof(double) * g_threadsPerKernel);

  switch (elemSize) {
  case 4:
    kernSweepSquares<<<g_blocksPerKernel, g_threadsPerBlock>>>
      ((float*)elems, buckets, subFromI, n);
    break;
  case 8:
    kernSweepSquares<<<g_blocksPerKernel, g_threadsPerBlock>>>
      ((double*)elems, buckets, subFromI, n);
    break;
  }

  const int nThreads = iMin(g_maxThreadsPerBlock, g_threadsPerKernel/4);
  kernSweepSums<<<1, nThreads>>>(buckets, g_threadsPerKernel);
  kernSweepSums<<<1, 1>>>(buckets, nThreads);

  double froNorm2;
  cuDownload(&froNorm2, buckets, sizeof(double));
  CUCHECK;
  return sqrt(froNorm2);
}

} // end extern "C"
