// blas.c

#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include "acmi.h"

void blasGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
  assert (mA != mC && mB != mC);
  const int n = MatN(mA);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
              MatElements(mA), n, MatElements(mB), n, beta, MatElements(mC), n);
}

void blasGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
  assert(mA != mC);
  const int n = MatN(mA);

  for (int i = 0; i < n; ++i) {
    if (mB == mC) {
      cblas_sscal(n, beta, MatElements(mB) + i*n, 1);
      cblas_saxpy(n, alpha, MatElements(mA) + i*n, 1, MatElements(mB) + i*n, 1);
    } else {
      memset(MatElements(mC) + i*n, 0, n*sizeof(float));
      cblas_saxpy(n, alpha, MatElements(mA) + i*n, 1, MatElements(mC) + i*n, 1);
      cblas_saxpy(n, beta, MatElements(mB) + i*n, 1, MatElements(mC) + i*n, 1);
    }
  }
}

float blasNorm(Mat mA) {
  const int n = MatN(mA);
  return LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, MatElements(mA), n);
}
