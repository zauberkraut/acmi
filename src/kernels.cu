/* kernels.cu

   Custom ACMI CUDA kernels. */

#include "acmi.h"

namespace {

/* Kernel parameters. */
int g_blocksPerVector, g_threadsPerVectorBlock, g_threadsPerVector,
    g_blocksPerMatrix, g_threadsPerMatrixBlock, g_threadsPerMatrix;

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
kernAddId(const T alpha, T* a, const int n) {
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
  if (maxBlocksPerKernel > prop.maxGridSize[0]) {
    fatal("max blocks supported: %d", prop.maxGridSize[0]);
  }
  const int maxThreadsPerBlock = prop.maxThreadsPerBlock;

  const int64_t n2 = n*n;
  g_blocksPerVector = (n  + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
  g_blocksPerVector = iMin(maxBlocksPerKernel, g_blocksPerVector);
  g_blocksPerMatrix = (n2 + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
  g_blocksPerMatrix = iMin(maxBlocksPerKernel, g_blocksPerMatrix);
  g_threadsPerVectorBlock = iMin(maxThreadsPerBlock, n);
  g_threadsPerMatrixBlock = iMin(maxThreadsPerBlock, n2);
  g_threadsPerVector = g_blocksPerVector * g_threadsPerVectorBlock;
  g_threadsPerMatrix = g_blocksPerMatrix * g_threadsPerMatrixBlock;

  debug("max  blocks/kernel: %d\n"
        "max threads/block : %d\n"
        "     blocks/vector: %d\n"
        "     blocks/matrix: %d\n"
        "    threads/vector: %d\n"
        "    threads/matrix: %d", maxBlocksPerKernel, maxThreadsPerBlock,
        g_blocksPerVector, g_blocksPerMatrix,
        g_threadsPerVector, g_threadsPerMatrix);
}

void cuShutDown() {
  debug("shutting down kernels");
}

/* Doubles matrix precision. */
void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  kernCopy<<<g_blocksPerMatrix, g_threadsPerMatrixBlock>>>
    ((double*)dst, (const float*)src, n2);
}

/* Adds alpha*I to the nxn matrix backed by the device array elems. */
void cuAddId(double alpha, void* elems, int n, int elemSize) {
  switch (elemSize) {
  case 4:
    kernAddId<<<g_blocksPerVector, g_threadsPerVectorBlock>>>
      ((float)alpha, (float*)elems, n);
    break;
  case 8:
    kernAddId<<<g_blocksPerVector, g_threadsPerVectorBlock>>>
      (alpha, (double*)elems, n);
    break;
  }
}

} // end extern "C"
