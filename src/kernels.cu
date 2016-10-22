/* kernels.cu

   Custom ACMI CUDA kernels. */

#include "acmi.h"

namespace {

enum { SWEEP_FACTOR = 4 };

/* Kernel parameters. */
int g_blocksPerVector, g_threadsPerVectorBlock,
    g_blocksPerSweep,  g_threadsPerSweepBlock,
    g_blocksPerMatrix, g_threadsPerMatrixBlock;
double* g_buckets;
int g_bucketsLen;

/* Copies elements from one nxn matrix to another, converting them
   to the precision of the destination matrix. */
template<typename T, typename U> __global__ void
kernCopy(T* dst, const U* src, const int64_t n2) {
  const int offset = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  const T* const end = dst + n2;
  src += offset;
  dst += offset;

  for (; dst < end; dst += stride, src += stride) {
    *dst = *src;
  }
}

template<typename T> __global__ void
kernAddId(const T alpha, T* a, const int n) {
  const T* const end = a + n*n;
  a += (blockIdx.x*blockDim.x + threadIdx.x)*(n + 1);
  const int stride = gridDim.x*blockDim.x*(n + 1);

  for (; a < end; a += stride) {
    *a += alpha;
  }
}

/* Sweeps sums of matrix elements into buckets. */
template<typename T> __global__ void
kernSweepSums(const T* a, const int64_t len, const int pitch,
              T* const buckets, const int bucketsLen) {
  const int offset = blockIdx.x*blockDim.x + threadIdx.x;
  const T* const end = a + len;
  a += offset*pitch;
  const int stride = gridDim.x*blockDim.x*pitch;

  if (offset < bucketsLen) {
    T partialSum = 0.;
    for (; a < end; a += stride) {
      partialSum += *a;
    }
    buckets[offset] = partialSum;
  }
}

} // end anonymous namespace

extern "C" {

/* Sets up kernel parameters. */
void cuSetUp(const int maxBlocksPerKernel, const int n) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // assume using first device
  debug("setting up kernels on %s", prop.name);
  if (maxBlocksPerKernel > prop.maxGridSize[0]) {
    fatal("max blocks supported: %d", prop.maxGridSize[0]);
  }
  const int maxThreadsPerBlock = prop.maxThreadsPerBlock;

  const int64_t n2 = n*n;
  g_bucketsLen = (n + SWEEP_FACTOR - 1)/SWEEP_FACTOR;
  g_buckets = (double*)cuMalloc(sizeof(double)*g_bucketsLen);

  g_blocksPerVector = (n  + maxThreadsPerBlock - 1) /
                      maxThreadsPerBlock;
  g_blocksPerVector = iMin(maxBlocksPerKernel, g_blocksPerVector);
  g_blocksPerSweep  = (g_bucketsLen  + maxThreadsPerBlock - 1) /
                      maxThreadsPerBlock;
  g_blocksPerSweep  = iMin(maxBlocksPerKernel, g_blocksPerSweep);
  g_blocksPerMatrix = (n2 + maxThreadsPerBlock - 1) /
                      maxThreadsPerBlock;
  g_blocksPerMatrix = iMin(maxBlocksPerKernel, g_blocksPerMatrix);
  g_threadsPerVectorBlock = iMin(maxThreadsPerBlock, n);
  g_threadsPerSweepBlock  = iMin(maxThreadsPerBlock, g_bucketsLen);
  g_threadsPerMatrixBlock = iMin(maxThreadsPerBlock, n2);
  int threadsPerVector = g_blocksPerVector*g_threadsPerVectorBlock,
      threadsPerMatrix = g_blocksPerMatrix*g_threadsPerMatrixBlock;

  debug("max  blocks/kernel: %d\n"
        "max threads/block : %d\n"
        "     blocks/matrix: %d\n"
        "    threads/matrix: %d\n"
        "     blocks/vector: %d\n"
        "    threads/vector: %d",
        maxBlocksPerKernel, maxThreadsPerBlock, g_blocksPerMatrix,
        threadsPerMatrix, g_blocksPerVector, threadsPerVector);
}

/* Cleans up kernel environment. */
void cuShutDown() {
  debug("shutting down kernels");
  cuFree(g_buckets);
}

/* Doubles matrix precision. */
void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  kernCopy<<<g_blocksPerMatrix, g_threadsPerMatrixBlock>>>
    ((double*)dst, (const float*)src, n2);
}

/* Adds alpha*I to the matrix backed by the device array elems. */
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

/* Computes a matrix trace by sweeping sums. */
double cuTrace(void* elems, int n, int elemSize) {
  const int64_t n2 = n*n;
  double trace = NAN;

  switch (elemSize) {
    float trace32;

  case 4:
    kernSweepSums<<<g_blocksPerSweep, g_threadsPerSweepBlock>>>
      ((float*)elems, n2, n + 1, (float*)g_buckets, g_bucketsLen);
    kernSweepSums<<<1, 1>>>
      ((float*)g_buckets, g_bucketsLen, 1, (float*)g_buckets, 1);
    cuDownload(&trace32, g_buckets, sizeof(float));
    trace = trace32;
    break;
  case 8:
    kernSweepSums<<<g_blocksPerSweep, g_threadsPerSweepBlock>>>
      ((double*)elems, n2, n + 1, (double*)g_buckets, g_bucketsLen);
    kernSweepSums<<<1, 1>>>
      ((double*)g_buckets, g_bucketsLen, 1, (double*)g_buckets, 1);
    cuDownload(&trace, g_buckets, sizeof(double));
    break;
  }

  return trace;
}

} // end extern "C"
