/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <cassert>
#include <stdint.h>
#include "acmi.h"

namespace {
  int g_maxBlocksPerGrid, g_maxThreadsPerBlock,
      g_blocksPerKernel,  g_threadsPerBlock,
      g_threadsPerKernel;

__global__ void kern32to64(double* dst, const float* src,
                                  const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = src[i];
  }
}

__global__ void kern32SetDiag(float* elems, float alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

__global__ void kern64SetDiag(double* elems, double alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

__global__ void kern32AddDiag(float* a, const float alpha,
                                     const int n) {
  for (int i = 0; i < n; i++) {
    const int j = i*n + i;
    a[j] = a[j] + alpha;
  }
}

__global__ void kern64AddDiag(double* a, const double alpha,
                                     const int n) {
  for (int i = 0; i < n; i++) {
    const int j = i*n + i;
    a[j] = a[j] + alpha;
  }
}

template<typename T> __global__ void
kernSweepSquares(const T* const a, double* const buckets,
                 const bool subFromI, const int n) {
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

__global__ void kernSweepSums(double* const buckets, const int bucketsLen) {
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
  g_threadsPerBlock = n2 > g_maxThreadsPerBlock ? g_maxThreadsPerBlock : n2;
  g_threadsPerKernel = g_threadsPerBlock * g_blocksPerKernel;
  debug("  max blocks/grid: %d\nmax threads/block: %d\n"
        "    blocks/kernel: %d\n    threads/block: %d\n"
        "   threads/kernel: %d", g_maxBlocksPerGrid, g_maxThreadsPerBlock,
        g_blocksPerKernel, g_threadsPerBlock, g_threadsPerKernel);
}

void cuShutDown() {
  debug("shutting down kernels");
}

void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  switch (srcElemSize) {
  case 4: kern32to64<<<1, 1>>>((double*)dst, (const float*)src, n2); break;
  case 8: /* WIP */; break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

void cuSetDiag(void* elems, double alpha, int n, int elemSize) {
  switch (elemSize) {
  case 4: kern32SetDiag<<<1, 1>>>((float*)elems, alpha, n);  break;
  case 8: kern64SetDiag<<<1, 1>>>((double*)elems, alpha, n); break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

void cuAddDiag(void* elems, double alpha, int n, int elemSize) {
  switch (elemSize) {
  case 4: kern32AddDiag<<<1, 1>>>((float*)elems,  alpha, n); break;
  case 8: kern64AddDiag<<<1, 1>>>((double*)elems, alpha, n); break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

// TODO: test manual gemm
// TODO: test combined kernels
// TODO: confirm thread count numbers
// TODO: sweep sums opt
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

  kernSweepSums<<<1, 1>>>(buckets, g_threadsPerKernel);

  double froNorm2;
  cuDownload(&froNorm2, buckets, sizeof(double));
  return sqrt(froNorm2);
}

void cuHgeam(float alpha, void* a, float beta, void* b, void* c, int64_t n2) {
  assert(false);
}

} // end extern "C"
