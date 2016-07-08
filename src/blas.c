// blas.c

#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include "invmat.h"

void blasGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
  if (mA == mC || mB == mC) {
    fatal("gemm operand may not be both source and target");
  }
  const int n = mA->n;
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
              mA->p, n, mB->p, n, beta, mC->p, n);
}

void blasGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
  if (mA == mC) {
    fatal("first geam operand may not be both source and target");
  }
  const int n = mA->n;

  for (int i = 0; i < n; ++i) {
    if (mB == mC) {
      cblas_sscal(n, beta, mB->p + i*n, 1);
      cblas_saxpy(n, alpha, mA->p + i*n, 1, mB->p + i*n, 1);
    } else {
      memset(mC->p + i*n, 0, n*sizeof(float));
      cblas_saxpy(n, alpha, mA->p + i*n, 1, mC->p + i*n, 1);
      cblas_saxpy(n, beta, mB->p + i*n, 1, mC->p + i*n, 1);
    }
  }
}

float blasNorm(Mat mA) {
  const int n = mA->n;
  return LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, mA->p, n);
}
