/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <cassert>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

static __global__ void kern16to32(float* dst, const __half* src,
                                  const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = __half2float(src[i]);
  }
}

static __global__ void kern32to64(double* dst, const float* src,
                                  const int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    dst[i] = src[i];
  }
}

void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2) {
  switch (srcElemSize) {
  case 2: kern16to32<<<1, 1>>>((float*)dst, (const __half*)src, n2); break;
  case 4: kern32to64<<<1, 1>>>((double*)dst, (const float*)src, n2); break;
  case 8: /* WIP */; break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

static __global__ void kern16SetDiag(__half* elems, float alpha, int n) {
  __half a = __float2half(alpha);
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = a;
  }
}

static __global__ void kern32SetDiag(float* elems, float alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

static __global__ void kern64SetDiag(double* elems, double alpha, int n) {
  for (int i = 0; i < n; i++) {
    elems[i*n + i] = alpha;
  }
}

void cuSetDiag(void* elems, double alpha, int n, int elemSize) {
  switch (elemSize) {
  case 2:
    kern16SetDiag<<<1, 1>>>((__half*)elems, alpha, n);
    break;
  case 4: kern32SetDiag<<<1, 1>>>((float*)elems, alpha, n);  break;
  case 8: kern64SetDiag<<<1, 1>>>((double*)elems, alpha, n); break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

static __global__ void kern16AddDiag(__half* a, const float alpha,
                                     const int n) {
  for (int i = 0; i < n; i++) {
    const int j = i*n + i;
    a[j] = __float2half(__half2float(a[j]) + alpha);
  }
}

static __global__ void kern32AddDiag(float* a, const float alpha,
                                     const int n) {
  for (int i = 0; i < n; i++) {
    const int j = i*n + i;
    a[j] = a[j] + alpha;
  }
}

static __global__ void kern64AddDiag(double* a, const double alpha,
                                     const int n) {
  for (int i = 0; i < n; i++) {
    const int j = i*n + i;
    a[j] = a[j] + alpha;
  }
}

void cuAddDiag(void* elems, double alpha, int n, int elemSize) {
  switch (elemSize) {
  case 2: kern16AddDiag<<<1, 1>>>((__half*)elems, alpha, n); break;
  case 4: kern32AddDiag<<<1, 1>>>((float*)elems,  alpha, n); break;
  case 8: kern64AddDiag<<<1, 1>>>((double*)elems, alpha, n); break;
  }
  assert(cudaSuccess == cudaGetLastError());
}

__device__ double d_froNorm;

static __global__ void kern16Norm(const __half* a, const int64_t n2) {
  double sum = 0;
  for (int64_t i = 0; i < n2; i++) {
    double e = __half2float(a[i]);
    sum += e*e;
  }
  d_froNorm = sqrt(sum);
}

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

double cuNorm(void* elems, int64_t n2, int elemSize) {
  switch (elemSize) {
  case 2: kern16Norm<<<1, 1>>>((__half*)elems, n2); break;
  case 4: kern32Norm<<<1, 1>>>((float*)elems, n2);  break;
  case 8: kern64Norm<<<1, 1>>>((double*)elems, n2); break;
  }

  assert(cudaSuccess == cudaGetLastError());
  typeof(d_froNorm) froNorm;
  cudaMemcpyFromSymbol(&froNorm, d_froNorm, sizeof(froNorm), 0, cudaMemcpyDeviceToHost);
  return froNorm;
}

static __global__ void kern16NormSubFromI(__half* a, int n) {
  double sum = 0;
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < n; row++) {
      int i = col*n + row;
      double e = (col == row) - __half2float(a[i]);
      sum += e*e;
    }
  }
  d_froNorm = sqrt(sum);
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

double cuNormSubFromI(void* elems, int n, int elemSize) {
  switch (elemSize) {
  case 2: kern16NormSubFromI<<<1, 1>>>((__half*)elems, n); break;
  case 4: kern32NormSubFromI<<<1, 1>>>((float*)elems, n);  break;
  case 8: kern64NormSubFromI<<<1, 1>>>((double*)elems, n); break;
  }

  assert(cudaSuccess == cudaGetLastError());
  typeof(d_froNorm) froNorm;
  cudaMemcpyFromSymbol(&froNorm, d_froNorm, sizeof(froNorm), 0, cudaMemcpyDeviceToHost);
  return froNorm;
}

static __global__ void kernHgeam(float alpha, __half* a, float beta, __half* b,
                                 __half* c, int64_t n2) {
  for (int64_t i = 0; i < n2; i++) {
    c[i] = __float2half(alpha * __half2float(a[i]) + beta * __half2float(b[i]));
  }
}

void cuHgeam(float alpha, void* a, float beta, void* b, void* c, int64_t n2) {
  kernHgeam<<<1, 1>>>(alpha, (__half*)a, beta, (__half*)b, (__half*)c, n2);
  assert(cudaSuccess == cudaGetLastError());
}

} // end extern "C"
