/* blas.c
 
   ACMI linear algebra operations implemented using (cu)BLAS. */

#include <cublas_v2.h>
#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include "acmi.h"

static cublasHandle_t g_cublasHandle = 0;

void initCublas() {
  debug("initializing cuBLAS");
  if (cublasCreate(&g_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    fatal("couldn't open cuBLAS handle");
  }
}

void shutDownCublas() {
  debug("shutting down cuBLAS");
  if (g_cublasHandle) cublasDestroy(g_cublasHandle);
}

/* C = alpha*A*B + beta*C */
void gemm(double alpha, Mat mA, Mat mB, double beta, Mat mC) {
  assert (mA != mC && mB != mC);
  const int n = MatN(mA);

  if (MatDev(mA)) { // matrix elements reside in device memory
    if (MatDouble(mA)) { // entries are double-precision
      cublasDgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                  MatElems(mA), n, MatElems(mB), n, &beta, MatElems(mC), n);
    } else { // entries are single-precision
      float a = alpha; float b = beta;
      cublasSgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a,
                  MatElems(mA), n, MatElems(mB), n, &b, MatElems(mC), n);
    }
  } else { // matrix elements reside in host memory
    if (MatDouble(mA)) {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
                  MatElems(mA), n, MatElems(mB), n, beta, MatElems(mC), n);
    } else {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
                  MatElems(mA), n, MatElems(mB), n, beta, MatElems(mC), n);
    }
  }
}

void gemmT(double alpha, Mat mA, Mat mB, double beta, Mat mC) {
  assert (mA != mC && mB != mC);
  const int n = MatN(mA);

  if (MatDev(mA)) {
    if (MatDouble(mA)) {
      cublasDgemm(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha,
                  MatElems(mA), n, MatElems(mB), n, &beta, MatElems(mC), n);
    } else {
      float a = alpha; float b = beta;
      cublasSgemm(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &a,
                  MatElems(mA), n, MatElems(mB), n, &b, MatElems(mC), n);
    }
  } else {
    if (MatDouble(mA)) {
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, alpha,
                  MatElems(mA), n, MatElems(mB), n, beta, MatElems(mC), n);
    } else {
      cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, alpha,
                  MatElems(mA), n, MatElems(mB), n, beta, MatElems(mC), n);
    }
  }
}

/* mB == mC => C = alpha*A + beta*C
   otherwise   C = alpha*A + beta*B */
void geam(double alpha, Mat mA, double beta, Mat mB, Mat mC) {
  assert(mA != mC);
  const int n = MatN(mA);

  if (MatDev(mA)) {
    if (MatDouble(mA)) {
      cublasDgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha,
                  MatElems(mA), n, &beta, MatElems(mB), n, MatElems(mC), n);
    } else {
      float a = alpha; float b = beta;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &a,
                  MatElems(mA), n, &b, MatElems(mB), n, MatElems(mC), n);
    }
  } else { // software
    for (int col = 0; col < n; ++col) {
      if (mB == mC) {
        if (MatDouble(mA)) {
          cblas_dscal(n, beta, MatCol(mB, col), 1);
          cblas_daxpy(n, alpha, MatCol(mA, col), 1, MatCol(mB, col), 1);
        } else {
          cblas_sscal(n, beta, MatCol(mB, col), 1);
          cblas_saxpy(n, alpha, MatCol(mA, col), 1, MatCol(mB, col), 1);
        }
      } else {
        memset(MatCol(mC, col), 0, MatPitch(mC));
        if (MatDouble(mA)) {
          cblas_daxpy(n, alpha, MatCol(mA, col), 1, MatCol(mC, col), 1);
          cblas_daxpy(n, beta, MatCol(mB, col), 1, MatCol(mC, col), 1);
        } else {
          cblas_saxpy(n, alpha, MatCol(mA, col), 1, MatCol(mC, col), 1);
          cblas_saxpy(n, beta, MatCol(mB, col), 1, MatCol(mC, col), 1);
        }
      }
    }
  }
}

/* Computes the Frobenius norm of a matrix. */
double norm(Mat mA) {
  const int n = MatN(mA);
  double froNorm;

  if (MatDev(mA)) {
    void* colVector = cuMalloc(n*MatElemSize(mA));

    // set output parameter location to device
    cublasSetPointerMode(g_cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    // compute the column Euclidean norms
    for (int col = 0; col < n; ++col) {
      if (MatDouble(mA)) {
        cublasDnrm2(g_cublasHandle, n, MatCol(mA, col), 1,
                    (double*)colVector + col);
      } else {
        cublasSnrm2(g_cublasHandle, n, MatCol(mA, col), 1,
                    (float*)colVector + col);
      }
    }

    // set output parameter location to host
    cublasSetPointerMode(g_cublasHandle, CUBLAS_POINTER_MODE_HOST);
    // compute the Euclidean norm of the column norms
    if (MatDouble(mA)) {
      cublasDnrm2(g_cublasHandle, n, (double*)colVector, 1, &froNorm);
    } else {
      float froNorm32;
      cublasSnrm2(g_cublasHandle, n, (float*)colVector, 1, &froNorm32);
      froNorm = froNorm32;
    }

    cuFree(colVector);
  } else { // software
    froNorm = MatDouble(mA) ?
      LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n) :
      LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n);
  }

  return froNorm;
}

// overwrites input matrix A!
double luInvert(Mat mA, Mat mR) {
  assert(MatDev(mA) && MatDev(mR));

  const int n = MatN(mA);

  // these cuBLAS functions work on arrays of matrices; we only want to invert
  // one matrix, so we allocate a singleton array of element pointers
  void** devAp = cuMalloc(sizeof(void*));
  void** devRp = cuMalloc(sizeof(void*));
  int* devPivots = cuMalloc(n*sizeof(int));
  int* devInfo = cuMalloc(sizeof(int));
  int hostInfo;

  void* hostAp = MatElems(mA);
  void* hostRp = MatElems(mR);
  cuUpload(devAp, &hostAp, sizeof(void*));
  cuUpload(devRp, &hostRp, sizeof(void*));

  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  // prerequisite LU factorization
  if ((MatElemSize(mA) == 4 ?
       cublasSgetrfBatched(g_cublasHandle, n, (float**)devAp, n, devPivots,
                           devInfo, 1) :
       cublasDgetrfBatched(g_cublasHandle, n, (double**)devAp, n, devPivots,
                           devInfo, 1))
      != CUBLAS_STATUS_SUCCESS) {
    fatal("LU factorization failed");
  }
  // inversion
  if ((MatElemSize(mA) == 4 ?
       cublasSgetriBatched(g_cublasHandle, n, (const float**)devAp, n,
       (const int*)devPivots, (float**)devRp, n, devInfo, 1) :
       cublasDgetriBatched(g_cublasHandle, n, (const double**)devAp, n,
       (const int*)devPivots, (double**)devRp, n, devInfo, 1))
      != CUBLAS_STATUS_SUCCESS) {
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

  // repurpose mA to compute the inversion error
  Mat mX = MatBuild(mA);
  gemm(-1, mA, mR, 0, mX); // X = -A*R
  MatClear(mA);
  for (int i = 0; i < n; ++i) { // A = I
    MatPut(mA, i, i, 1);
  }
  geam(1, mA, 1, mX, mX); // X = I - A*R
  double error = norm(mX); // || I - A*R ||
  MatFree(mX);

  debug("LU factorization and inversion completed after %g seconds "
        "with error measure %g", invTimeS, error);
  return error;
}
