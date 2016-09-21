/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" size_t cuMemAvail() {
  size_t free, total;
  assert(cudaSuccess == cudaMemGetInfo(&free, &total));
  return free;
}

extern "C" void* cuMalloc(size_t size) {
  void* p;
  assert(cudaSuccess == cudaMalloc(&p, size));
  return p;
}

extern "C" void cuFree(void* p) { assert(cudaSuccess == cudaFree(p)); }

extern "C" void cuClear(void* p, size_t size) {
  assert(cudaSuccess == cudaMemset(p, 0, size));
}

extern "C" void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  assert(cudaSuccess == cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice));
}

extern "C" void cuPin(void* p, size_t size) {
  assert(cudaSuccess == cudaHostRegister(p, size, cudaHostRegisterPortable));
}

extern "C" void cuUnpin(void* p) {
  assert(cudaSuccess == cudaHostUnregister(p));
}

extern "C" void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  assert(cudaSuccess == cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost));
}

static __global__ void kern32to16(__half* dst, const float* src, const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = __float2half(src[i]);
  }
}

extern "C" void cuDemote(uint16_t* dst, float* src, int64_t n2) {
  kern32to16<<<1, 1>>>((__half*)dst, src, n2);
  assert(cudaSuccess == cudaGetLastError());
}

static __global__ void kern16to32(float* dst, const __half* src, const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = __half2float(src[i]);
  }
}

static __global__ void kern32to64(double* dst, const float* src, const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = src[i];
  }
}

extern "C" void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  switch (srcElemSize) {
  case 2: kern16to32<<<1, 1>>>((float*)dst, (const __half*)src, n2); break;
  case 4: kern32to64<<<1, 1>>>((double*)dst, (const float*)src, n2); break;
  case 8: /* WIP */; break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

static __global__ void kernSetDiag32(float* elems, double alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

static __global__ void kernSetDiag64(double* elems, double alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

extern "C" void cuSetDiag(void* elems, double alpha, int n, int elemSize) {
  8 == elemSize ? kernSetDiag64<<<1, 1>>>((double*)elems, alpha, n)
                : kernSetDiag32<<<1, 1>>>((float*)elems, alpha, n);
  assert(cudaSuccess == cudaGetLastError());
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
  8 == elemSize ? kern64Norm<<<1, 1>>>((double*)elems, n2)
                : kern32Norm<<<1, 1>>>( (float*)elems, n2);
  assert(cudaSuccess == cudaGetLastError());
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
  8 == elemSize ? kern64NormSubFromI<<<1, 1>>>((double*)elems, n)
                : kern32NormSubFromI<<<1, 1>>>( (float*)elems, n);
  assert(cudaSuccess == cudaGetLastError());
  typeof(d_froNorm) froNorm;
  cudaMemcpyFromSymbol(&froNorm, d_froNorm, sizeof(froNorm), 0, cudaMemcpyDeviceToHost);
  return froNorm;
}

static __global__ void kern32Add3I(float* a, int n) {
  for (int i = 0; i < n; i++) {
    unsigned j = i*n + i;
    float e = a[j];
    a[j] = e + 3.0f;
  }
}

static __global__ void kern64Add3I(double* a, int n) {
  for (int i = 0; i < n; i++) {
    int j = i*n + i;
    double e = a[j];
    a[j] = e + 3.0;
  }
}

extern "C" void cuAdd3I(void* elems, int n, int elemSize) {
  8 == elemSize ? kern64Add3I<<<1, 1>>>((double*)elems, n)
                : kern32Add3I<<<1, 1>>>( (float*)elems, n);
  assert(cudaSuccess == cudaGetLastError());
}
