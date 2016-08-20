/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <assert.h>

extern "C" size_t cuMemAvail() {
  size_t free, total;
  assert(cudaMemGetInfo(&free, &total) == cudaSuccess);
  return free;
}

extern "C" void* cuMalloc(size_t size) {
  void* p;
  assert(cudaMalloc(&p, size) == cudaSuccess);
  return p;
}

extern "C" void cuFree(void* p) { assert(cudaFree(p) == cudaSuccess); }

extern "C" void cuClear(void* p, size_t size) {
  assert(cudaMemset(p, 0, size) == cudaSuccess);
}

extern "C" void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  assert(cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice) ==
         cudaSuccess);
}

extern "C" void cuPin(void* p, size_t size) {
  assert(cudaHostRegister(p, size, cudaHostRegisterPortable) == cudaSuccess);
}

extern "C" void cuUnpin(void* p) {
  assert(cudaHostUnregister(p) == cudaSuccess);
}

extern "C" void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  assert(cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
}

static __global__ void kernWiden(double* dst, const float* src, const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = src[i];
  }
}

extern "C" void cuWiden(double* dst, float* src, int64_t n2) {
  kernWiden<<<1, 1>>>(dst, src, n2);
  assert(cudaGetLastError() == cudaSuccess);
}

__device__ double d_froNorm;

static __global__ void kern32Norm(const float* a, const int64_t n2) {
  double sum = 0;
  for (int64_t i = 0; i < n2; i++) {
    double e = a[i];
    sum += e*e;
  }
  d_froNorm = sqrt(sum);
}

static __global__ void kern64Norm(const double* a, const int64_t n2) {
  double sum = 0;
  for (int64_t i = 0; i < n2; i++) {
    double e = a[i];
    sum += e*e;
  }
  d_froNorm = sqrt(sum);
}

extern "C" double cuNorm(void* elems, int64_t n2, int elemSize) {
  elemSize == 8 ? kern64Norm<<<1, 1>>>((double*)elems, n2)
                : kern32Norm<<<1, 1>>>( (float*)elems, n2);
  assert(cudaGetLastError() == cudaSuccess);
  typeof(d_froNorm) froNorm;
  cudaMemcpyFromSymbol(&froNorm, d_froNorm, sizeof(froNorm), 0, cudaMemcpyDeviceToHost);
  return froNorm;
}

static __global__ void kern32NormSubFromI(float* a, int n) {
  double sum = 0;
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < n; row++) {
      int i = col*n + row;
      double e = (col == row) - a[i];
      sum += e*e;
    }
  }
  d_froNorm = sqrt(sum);
}

static __global__ void kern64NormSubFromI(double* a, int n) {
  double sum = 0;
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < n; row++) {
      int i = col*n + row;
      double e = (col == row) - a[i];
      sum += e*e;
    }
  }
  d_froNorm = sqrt(sum);
}

extern "C" double cuNormSubFromI(void* elems, int n, int elemSize) {
  elemSize == 8 ? kern64NormSubFromI<<<1, 1>>>((double*)elems, n)
                : kern32NormSubFromI<<<1, 1>>>( (float*)elems, n);
  assert(cudaGetLastError() == cudaSuccess);
  typeof(d_froNorm) froNorm;
  cudaMemcpyFromSymbol(&froNorm, d_froNorm, sizeof(froNorm), 0, cudaMemcpyDeviceToHost);
  return froNorm;
}

static __global__ void kern32Add3I(float* a, int n) {
  for (int i = 0; i < n; ++i) {
    unsigned j = i*n + i;
    float e = a[j];
    a[j] = e + 3.0f;
  }
}

static __global__ void kern64Add3I(double* a, int n) {
  for (int i = 0; i < n; ++i) {
    int j = i*n + i;
    double e = a[j];
    a[j] = e + 3.0;
  }
}

extern "C" void cuAdd3I(void* elems, int n, int elemSize) {
  elemSize == 8 ? kern64Add3I<<<1, 1>>>((double*)elems, n)
                : kern32Add3I<<<1, 1>>>( (float*)elems, n);
  assert(cudaGetLastError() == cudaSuccess);
}
