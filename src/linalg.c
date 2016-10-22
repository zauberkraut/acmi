/* linalg.c
 
   ACMI linear algebraic operations implemented using (cu)BLAS and
   CUDA kernels.
*/

#include <cublas_v2.h>
#include <openblas/cblas.h>
#include "acmi.h"

static cublasHandle_t g_cublasHandle;

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

/* mT <- alpha*mA^T */
void transpose(double alpha, Mat mA, Mat mT) {
  const int n = MatN(mA);
  const void* const a = MatElems(mA);
  void* const t = MatElems(mT);
  const bool dev = MatDev(mA);
  const double beta = 0;

  switch (MatElemSize(mA)) {
  case 4:
    if (dev) {
      float alpha32 = alpha;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                  &alpha32, a, n, (float*)&beta, t, n, t, n);
    } else {
      cblas_somatcopy(CblasColMajor, CblasTrans, n, n, alpha, a, n,
                      t, n);
    }
    break;

  case 8:
    if (dev) {
      cublasDgeam(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                  &alpha, a, n, &beta, t, n, t, n);
    } else {
      cblas_domatcopy(CblasColMajor, CblasTrans, n, n, alpha, a, n,
                      t, n);
    }
    break;
  }
}

/* C <- alpha*A*B + beta*C */
void gemm(double alpha, Mat mA, Mat mB, double beta, Mat mC) {
  const int n = MatN(mA);
  const void* const a = MatElems(mA);
  const void* const b = MatElems(mB);
  void* const c = MatElems(mC);
  const bool dev = MatDev(mA);

  switch (MatElemSize(mA)) {
  case 4:
    if (dev) {
      float alpha32 = alpha, beta32 = beta;
      cublasSgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                  &alpha32, a, n, b, n, &beta32, c, n);
    } else {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n,
                  n, alpha, a, n, b, n, beta, c, n);
    }
    break;

  case 8:
    if (dev) {
      cublasDgemm(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                  &alpha, a, n, b, n, &beta, c, n);
    } else {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n,
                  n, alpha, a, n, b, n, beta, c, n);
    }
    break;
  }
}

/* mB = mC   => C <- alpha*A + beta*C
   otherwise    C <- alpha*A + beta*B */
void geam(double alpha, Mat mA, double beta, Mat mB, Mat mC) {
  const int n = MatN(mA);
  const int n2 = MatN2(mA);
  const void* const a = MatElems(mA);
  const void* const b = MatElems(mB);
  void* const c = MatElems(mC);
  const bool dev = MatDev(mA);

  switch (MatElemSize(mA)) {
  case 4:
    if (dev) {
      float alpha32 = alpha, beta32 = beta;
      cublasSgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                  &alpha32, a, n, &beta32, b, n, c, n);
    } else {
      if (b == c) {
        cblas_sscal(n2, beta, c, 1);
      } else {
        memset(c, 0, MatSize(mC));
        cblas_saxpy(n2, beta, b, 1, c, 1);
      }
      cblas_saxpy(n2, alpha, a, 1, c, 1);
    }
    break;

  case 8:
    if (dev) {
      cublasDgeam(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                  &alpha, a, n, &beta, b, n, c, n);
    } else {
      if (b == c) {
        cblas_dscal(n2, beta, c, 1);
      } else {
        memset(c, 0, MatSize(mC));
        cblas_daxpy(n2, beta, b, 1, c, 1);
      }
      cblas_daxpy(n2, alpha, a, 1, c, 1);
    }
    break;
  }
}

/* mA <- mA + alpha*I */
void addId(double alpha, Mat mA) {
  if (MatDev(mA)) {
    cuAddId(alpha, MatElems(mA), MatN(mA), MatElemSize(mA));
  } else for (int diag = 0; diag < MatN(mA); diag++) {
    /* This could be marginally sped up using *axpy with a 1xn
       vector of ones and a stride of n + 1 over the matrix, but
       it's not worth the trouble. */
    MatPut(mA, diag, diag, alpha + MatGet(mA, diag, diag));
  }
}

/* Computes the Frobenius norm of a matrix. */
double nrm2(Mat mA) {
  const int n2 = MatN2(mA);
  const void* a = MatElems(mA);
  const bool dev = MatDev(mA);
  double norm;

  switch (MatElemSize(mA)) {
  case 4:
    if (dev) {
      float norm32;
      cublasSnrm2(g_cublasHandle, n2, a, 1, (float*)&norm32);
      norm = norm32;
    } else {
      norm = cblas_snrm2(n2, a, 1);
    }
    break;

  case 8:
    if (dev) {
      cublasDnrm2(g_cublasHandle, n2, a, 1, (double*)&norm);
    } else {
      norm = cblas_dnrm2(n2, a, 1);
    }
    break;
  }

  return norm;
}

/* Computes the sum of the entries on the main diagonal. */
double trace(Mat mA) {
  double trace;

  if (MatDev(mA)) {
    trace = cuTrace(MatElems(mA), MatN(mA), MatElemSize(mA));
  } else {
    trace = 0.;
    for (int i = 0; i < MatN(mA); i++) {
      trace += MatGet(mA, i, i);
    }
  }

  return trace;
}

/* Computes the norm of (I - A). */
double minusIdNrm2(Mat mA) {
  addId(-1, mA); // sort of a hack, but it works very well
  double norm = nrm2(mA);
  addId(1, mA);
  return norm;
}
