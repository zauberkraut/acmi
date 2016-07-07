// cublas_la.c

#include <cublas_v2.h>
#include "invmat.h"

static cublasHandle_t cublHandle = 0;
static float* vX = 0;

void cublInit(unsigned n) {
  if (cublasCreate(&cublHandle) != CUBLAS_STATUS_SUCCESS) {
    fatal("couldn't open cuBLAS handle");
  }
  vX = cuMalloc(n*sizeof(float));
}

void cublShutDown() {
  if (cublHandle) cublasDestroy(cublHandle);
  if (vX) cuFree(vX);
}

void cublGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
  const int n = mA->n;
  cublasSgemm(cublHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, mA->p, n,
              mB->p, n, &beta, mC->p, n);
}

void cublGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
  const int n = mA->n;
  cublasSgeam(cublHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, mA->p, n,
              &beta, mB->p, n, mC->p, n);
}

float cublNorm(Mat mA) {
  const int n = mA->n;
  cublasSetPointerMode(cublHandle, CUBLAS_POINTER_MODE_DEVICE);
  for (int i = 0; i < n; ++i) {
    cublasSnrm2(cublHandle, n, mA->p + i*n, 1, vX + i);
  }
  float frobenius;
  cublasSetPointerMode(cublHandle, CUBLAS_POINTER_MODE_HOST);
  cublasSnrm2(cublHandle, n, vX, 1, &frobenius);
  return frobenius;
}

float luInvert(Mat mA, Mat mR) {
  const int n = mA->n;
  float** devAp = cuMalloc(sizeof(float*));
  float** devRp = cuMalloc(sizeof(float*));
  int* devPivots = cuMalloc(n*sizeof(int));
  int* devInfo = cuMalloc(sizeof(int));
  int hostInfo;

  cuUpload(devAp, &mA->p, sizeof(float*));
  cuUpload(devRp, &mR->p, sizeof(float*));

  if (cublasSgetrfBatched(cublHandle, n, devAp, n, devPivots, devInfo, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    fatal("LU factorization failed");
  }
  if (cublasSgetriBatched(cublHandle, n, (const float**)devAp, n,
      (const int*)devPivots, devRp, n, devInfo, 1) != CUBLAS_STATUS_SUCCESS) {
    fatal("inversion from LU factorization failed");
  }
  cuDownload(&hostInfo, devInfo, sizeof(int));
  if (hostInfo) {
    fatal("U from LU factorization is singular");
  }

  cuFree(devAp);
  cuFree(devRp);
  cuFree(devPivots);
  cuFree(devInfo);

  float error = measureAR(mA, mR);
  debug("LU factorization and inversion completed with error measure %g", error);
  return error;
}
