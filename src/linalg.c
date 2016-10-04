/* linalg.c
 
   ACMI linear algebraic operations implemented using (cu)BLAS and CUDA. */

#include <cublas_v2.h>
#include <openblas/cblas.h>
#include <lapacke.h>
#include "acmi.h"

static cublasHandle_t g_cublasHandle = 0;

void gpuSetUp(const int maxBlocksPerKernel, const int n) {
  debug("setting up cuBLAS");
  if (cublasCreate(&g_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    fatal("couldn't open cuBLAS handle");
  }
  cuSetUp(maxBlocksPerKernel, n);
}

void gpuShutDown() {
  debug("shutting down cuBLAS");
  if (g_cublasHandle) cublasDestroy(g_cublasHandle);
  cuShutDown();
}

/* mT = alpha*mA^T */
void transpose(double alpha, Mat mA, Mat mT) {
  const int n = MatN(mA);
  assert(n == MatN(mT));

  if (MatDev(mA)) {
    switch (MatElemSize(mA)) {
      union Elem a, beta;

    case 4:
      a.fp32 = alpha; beta.fp32 = 0;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &a.fp32,
                  MatElems(mA), n, &beta.fp32, MatElems(mT), n, MatElems(mT),
                  n);
      break;
    case 8:
      beta.fp64 = 0;
      cublasDgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha,
                  MatElems(mA), n, &beta.fp64, MatElems(mT), n, MatElems(mT),
                  n);
      break;
    }
  } else { // perform host memory transpose
    for (int row = 0; row < n; row++) {
      for (int col = row; col < n; col++) {
        if (row == col) {
          MatPut(mT, row, col, alpha*MatGet(mA, row, col));
        } else {
          double upper = alpha*MatGet(mA, row, col);
          double lower = alpha*MatGet(mA, col, row);
          MatPut(mT, row, col, lower);
          MatPut(mT, col, row, upper);
        }
      }
    }
  }
}

/* C = alpha*A*B + beta*C */
void gemm(double alpha, Mat mA, Mat mB, double beta, Mat mC) {
  assert (mA != mC && mB != mC);
  const int n = MatN(mA);

  if (MatDev(mA)) { // matrix elements reside in device memory
    switch (MatElemSize(mA)) {
      union Elem a, b;

    case 4:
      a.fp32 = alpha; b.fp32 = beta;
      cublasSgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a.fp32,
                  MatElems(mA), n, MatElems(mB), n, &b.fp32, MatElems(mC), n);
      break;
    case 8:
      cublasDgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                  MatElems(mA), n, MatElems(mB), n, &beta, MatElems(mC), n);
      break;
    }
  } else { // matrix elements reside in host memory
    if (8 == MatElemSize(mA)) {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
                  MatElems(mA), n, MatElems(mB), n, beta, MatElems(mC), n);
    } else {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
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
    switch (MatElemSize(mA)) {
      union Elem a, b;

    case 4:
      a.fp32 = alpha; b.fp32 = beta;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &a.fp32,
                  MatElems(mA), n, &b.fp32, MatElems(mB), n, MatElems(mC), n);
      break;
    case 8:
      cublasDgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha,
                  MatElems(mA), n, &beta, MatElems(mB), n, MatElems(mC), n);
      break;
    }
  } else { // software
    for (int col = 0; col < n; col++) {
      if (mB == mC) {
        if (8 == MatElemSize(mA)) {
          cblas_dscal(n, beta, MatCol(mB, col), 1);
          cblas_daxpy(n, alpha, MatCol(mA, col), 1, MatCol(mB, col), 1);
        } else {
          cblas_sscal(n, beta, MatCol(mB, col), 1);
          cblas_saxpy(n, alpha, MatCol(mA, col), 1, MatCol(mB, col), 1);
        }
      } else {
        memset(MatCol(mC, col), 0, MatPitch(mC));
        if (8 == MatElemSize(mA)) {
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

void setDiag(Mat mA, double alpha) {
  if (MatDev(mA)) {
    cuSetDiag(MatElems(mA), alpha, MatN(mA), MatElemSize(mA));
  } else {
    for (int i = 0; i < MatN(mA); i++) {
      MatPut(mA, i, i, alpha);
    }
  }
}

void addDiag(Mat mA, double alpha) {
  if (MatDev(mA)) {
    cuAddDiag(MatElems(mA), alpha, MatN(mA), MatElemSize(mA));
  } else {
    for (int diag = 0; diag < MatN(mA); diag++) {
      MatPut(mA, diag, diag, alpha + MatGet(mA, diag, diag));
    }
  }
}

/* Computes the Frobenius norm of a matrix. */
double froNorm(Mat mA, bool subFromI) {
  const int n = MatN(mA);
  double froNorm;

  if (MatDev(mA)) {
    froNorm = cuFroNorm(MatElems(mA), subFromI, n, MatElemSize(mA));
  } else { // software
    if (subFromI) { // horribly slow!
      double sum = 0;
      for (int row = 0; row < MatN(mA); row++) {
        for (int col = 0; col < MatN(mA); col++) {
          double e = (row == col) - MatGet(mA, row, col);
          sum += e*e;
        }
      }
      froNorm = sqrt(sum);
    } else {
      froNorm = 8 == MatElemSize(mA) ?
        LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n) :
        LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n);
    }
  }

  return froNorm;
}
