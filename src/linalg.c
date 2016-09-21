/* linalg.c
 
   ACMI linear algebraic operations implemented using (cu)BLAS and CUDA. */

#include <cublas_v2.h>
#include <openblas/cblas.h>
#include <lapacke.h>
#include "acmi.h"

static cublasHandle_t g_cublasHandle = 0;

void cublasInit() {
  debug("initializing cuBLAS");
  if (cublasCreate(&g_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    fatal("couldn't open cuBLAS handle");
  }
}

void cublasShutDown() {
  debug("shutting down cuBLAS");
  if (g_cublasHandle) cublasDestroy(g_cublasHandle);
}

/* mT = alpha*mA^T */
void transpose(double alpha, Mat mA, Mat mT) {
  const int n = MatN(mA);
  assert(n == MatN(mT));

  if (MatDev(mA)) {
    if (8 == MatElemSize(mA)) {
      double beta = 0;
      cublasDgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha,
                  MatElems(mA), n, &beta, MatElems(mT), n, MatElems(mT), n);
    } else {
      float alpha32 = alpha; float beta32 = 0;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha32,
                  MatElems(mA), n, &beta32, MatElems(mT), n, MatElems(mT), n);
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
    if (8 == MatElemSize(mA)) { // entries are double-precision
      cublasDgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                  MatElems(mA), n, MatElems(mB), n, &beta, MatElems(mC), n);
    } else { // entries are single-precision
      float alpha32 = alpha; float beta32 = beta;
      cublasSgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha32,
                  MatElems(mA), n, MatElems(mB), n, &beta32, MatElems(mC), n);
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
    if (8 == MatElemSize(mA)) {
      cublasDgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha,
                  MatElems(mA), n, &beta, MatElems(mB), n, MatElems(mC), n);
    } else {
      float alpha32 = alpha; float beta32 = beta;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha32,
                  MatElems(mA), n, &beta32, MatElems(mB), n, MatElems(mC), n);
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

/* Computes the Frobenius norm of a matrix. */
double norm(Mat mA) {
  const int n = MatN(mA);
  double froNorm;

  if (MatDev(mA)) {
    froNorm = cuNorm(MatElems(mA), MatN2(mA), MatElemSize(mA));
  } else { // software
    froNorm = 8 == MatElemSize(mA) ?
      LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n) :
      LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, MatElems(mA), n);
  }

  return froNorm;
}

double normSubFromI(Mat mA) {
  if (MatDev(mA)) {
    return cuNormSubFromI(MatElems(mA), MatN(mA), MatElemSize(mA));
  } else {
    double sum = 0;
    for (int row = 0; row < MatN(mA); row++) {
      for (int col = 0; col < MatN(mA); col++) {
        double e = (row == col) - MatGet(mA, row, col);
        sum += e*e;
      }
    }
    return sqrt(sum);
  }
}

void add3I(Mat mA) {
  if (MatDev(mA)) {
    cuAdd3I(MatElems(mA), MatN(mA), MatElemSize(mA));
  } else {
    for (int diag = 0; diag < MatN(mA); diag++) {
      MatPut(mA, diag, diag, 3 + MatGet(mA, diag, diag));
    }
  }
}
