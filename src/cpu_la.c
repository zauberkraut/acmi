// cpu_la.c

#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include "invmat.h"

uint64_t cpuTotalMatBytes = 0;

Mat cpuNewMat(unsigned n) {
  Mat m = emptyMat(n);
  m->device = false;
  m->p = malloc(m->size);

  cpuTotalMatBytes += m->size;
  debug("allocated %.3lf MiB to %dx%d matrix on host; %.3lf MiB total",
        mibibytes(m->size), n, n, mibibytes(cpuTotalMatBytes));
  return m;
}

void cpuFreeMat(Mat m) {
  free(m->p);
  m->p = 0;
  cpuTotalMatBytes -= m->size;
  debug("freed %.3lf MiB from %dx%d matrix on host; %.3lf MiB remain",
        mibibytes(m->size), m->n, m->n, mibibytes(cpuTotalMatBytes));
}

float cpuElem(Mat m, int row, int col) {
  bound(m, row, col);
  return m->p[col*m->n + row];
}

void cpuSetElem(Mat m, int row, int col, float e) {
  bound(m, row, col);
  m->p[col*m->n + row] = e;
}

void cpuGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
  if (mA == mC || mB == mC) {
    fatal("gemm operand may not be both source and target");
  }
  const int n = mA->n;
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
              mA->p, n, mB->p, n, beta, mC->p, n);
}

void cpuGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
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

float cpuNorm(Mat mA) {
  const int n = mA->n;
  return LAPACKE_slange(LAPACK_COL_MAJOR, 'F', n, n, mA->p, n);
}
