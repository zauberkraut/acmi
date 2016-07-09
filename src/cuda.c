// cuda.c

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "acmi.h"

static cublasHandle_t g_cublHandle = 0;
static float* g_cublVector = 0;

static void cuCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    fatal("%s:%d CUDA error %d: %s", err,
          cudaGetErrorString(err));
  }
}

void* cuMalloc(size_t size) {
  void* p;
  cuCheck(cudaMalloc(&p, size));
  return p;
}

void cuFree(void* p) {
  cuCheck(cudaFree(p));
}

void cuClear(void* p, size_t size) {
  cuCheck(cudaMemset(p, 0, size));
}

void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  cuCheck(cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice));
}

void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  cuCheck(cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost));
}

void cublInit(int n) {
  debug("initializing cuBLAS");
  if (cublasCreate(&g_cublHandle) != CUBLAS_STATUS_SUCCESS) {
    fatal("couldn't open cuBLAS handle");
  }
  g_cublVector = cuMalloc(n*sizeof(float));
}

void cublShutDown() {
  debug("shutting down cuBLAS");
  if (g_cublHandle) cublasDestroy(g_cublHandle);
  if (g_cublVector) cuFree(g_cublVector);
}

void cublGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
  const int n = MatN(mA);
  cublasSgemm(g_cublHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
              MatElements(mA), n, MatElements(mB), n, &beta, MatElements(mC),
              n);
}

void cublGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
  const int n = MatN(mA);
  cublasSgeam(g_cublHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha,
              MatElements(mA), n, &beta, MatElements(mB), n, MatElements(mC),
              n);
}

float cublNorm(Mat mA) {
  const int n = MatN(mA);
  cublasSetPointerMode(g_cublHandle, CUBLAS_POINTER_MODE_DEVICE);
  for (int i = 0; i < n; ++i) {
    cublasSnrm2(g_cublHandle, n, MatElements(mA) + i*n, 1, g_cublVector + i);
  }
  float frobenius;
  cublasSetPointerMode(g_cublHandle, CUBLAS_POINTER_MODE_HOST);
  cublasSnrm2(g_cublHandle, n, g_cublVector, 1, &frobenius);
  return frobenius;
}

// overwrites input matrix A!
float luInvert(Mat mA, Mat mR) {
  const int n = MatN(mA);
  float** devAp = cuMalloc(sizeof(float*));
  float** devRp = cuMalloc(sizeof(float*));
  int* devPivots = cuMalloc(n*sizeof(int));
  int* devInfo = cuMalloc(sizeof(int));
  int hostInfo;

  float* hostAp = MatElements(mA);
  float* hostRp = MatElements(mR);
  cuUpload(devAp, &hostAp, sizeof(float*));
  cuUpload(devRp, &hostRp, sizeof(float*));

  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  if (cublasSgetrfBatched(g_cublHandle, n, devAp, n, devPivots, devInfo, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    fatal("LU factorization failed");
  }
  if (cublasSgetriBatched(g_cublHandle, n, (const float**)devAp, n,
      (const int*)devPivots, devRp, n, devInfo, 1) != CUBLAS_STATUS_SUCCESS) {
    fatal("inversion from LU factorization failed");
  }
  cuDownload(&hostInfo, devInfo, sizeof(int));
  if (hostInfo) {
    fatal("U from LU factorization is singular");
  }

  clock_gettime(CLOCK_MONOTONIC, &endTime);
  double invTimeS = endTime.tv_sec - startTime.tv_sec +
    (endTime.tv_nsec - startTime.tv_nsec)/1.e9;

  cuFree(devAp);
  cuFree(devRp);
  cuFree(devPivots);
  cuFree(devInfo);

  Mat mX = MatNew(n, true);
  cublGemm(-1, mA, mR, 0, mX);
  MatClear(mA);
  for (int i = 0; i < n; ++i) {
    MatPut(mA, i, i, 1);
  }
  cublGeam(1, mA, 1, mX, mX);
  float error = cublNorm(mX);
  MatFree(mX);

  debug("LU factorization and inversion completed after %g seconds "
        "with error measure %g", invTimeS, error);
  return error;
}
