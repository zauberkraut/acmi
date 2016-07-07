// la.c

#include "invmat.h"

static Impl impl;
static Mat mI = 0;
static Mat mAR = 0;
static Mat mX = 0;

Mat emptyMat(unsigned n) {
  Mat m = malloc(sizeof(Mat_));
  memset(m, 0, sizeof(Mat_));
  m->n = n;
  m->n2 = (uint64_t)n * n;
  m->size = m->n2 * sizeof(float);
  return m;
}

Mat zeroMat(unsigned n) {
  Mat m = newMat(n);
  clearMat(m);
  return m;
}

void clearMat(Mat m) {
  if (m->device) cuClear(m->p, m->size);
  else           memset(m->p, 0, m->size);
}

void freeMat(Mat m) {
  if (!m || !m->p) {
    warn("attempted to free null or empty matrix");
  } else {
    if (m->device) {
      cuFreeMat(m);
    } else {
      cpuFreeMat(m);
    }
    free(m);
  }
}

void bound(Mat m, int row, int col) {
  if (row < 0 || row > m->n || col < 0 || col > m->n) {
    fatal("attempted to access element (%d, %d) of a %dx%d matrix", row, col,
          m->n, m->n);
  }
}

Mat (*newMat)(unsigned n) = 0;
float (*elem)(Mat m, int row, int col) = 0;
void (*setElem)(Mat m, int row, int col, float e) = 0;
static void (*gemm)(float alpha, Mat mA, Mat mB, float beta, Mat mC) = 0;
static void (*geam)(float alpha, Mat mA, float beta, Mat mB, Mat mC) = 0;
static float (*norm)(Mat mA) = 0;
static void (*invStep)(Mat mR) = 0;

static void cubicInvStep(Mat mR) {
  gemm(1, mR, mAR, 0, mX);
  geam(-3, mX, 3, mR, mR);
  gemm(1, mX, mAR, 1, mR);
}

static void quadInvStep(Mat mR) {
  gemm(1, mR, mAR, 0, mX);
  geam(-1, mX, 2, mR, mR);
}

void init(unsigned n, Impl impl_, bool quadConv) {
  impl = impl_;
  const char* implStr = 0;
  switch (impl) {
    case CPU_IMPL:
      implStr = "CPU OpenBLAS";
      newMat = cpuNewMat;
      elem = cpuElem;
      setElem = cpuSetElem;
      gemm = cpuGemm;
      geam = cpuGeam;
      norm = cpuNorm;
      break;
    case CUBLAS_IMPL:
    case LU_IMPL:
      implStr = "Nvidia cuBLAS";
      cublInit(n);
      newMat = cuNewMat;
      elem = cuElem;
      setElem = cuSetElem;
      gemm = cublGemm;
      geam = cublGeam;
      norm = cublNorm;
      break;
    case CUDA_IMPL:
      implStr = "Custom CUDA";
      newMat = cuNewMat;
      elem = cuElem;
      setElem = cuSetElem;
      gemm = cuGemm;
      geam = cuGeam;
      norm = cuNorm;
      break;
  }

  mI = zeroMat(n);
  for (int i = 0; i < n; ++i) {
    setElem(mI, i, i, 1);
  }
  mAR = newMat(n);
  mX = newMat(n);

  invStep = quadConv ? quadInvStep : cubicInvStep;

  debug("initialized %s engine; using %s-convergent algorithm", implStr,
        quadConv ? "quadratically" : "cubically");
}

void shutDown() {
  if (mI)  freeMat(mI);
  if (mAR) freeMat(mAR);
  if (mX)  freeMat(mX);

  switch (impl) {
    case CPU_IMPL:
      break;
    case CUBLAS_IMPL:
    case LU_IMPL:
      cublShutDown();
      break;
    case CUDA_IMPL:
      break;
  }

  if (cpuTotalMatBytes) {
    warn("%.3lf MiB remain allocated to matrices on host",
         mibibytes(cpuTotalMatBytes));
  }
  if (cuTotalMatBytes) {
    warn("%.3lf MiB remain allocated to matrices on device",
         mibibytes(cuTotalMatBytes));
  }
}

static void computeR0(Mat mA, Mat mR) {
  const float normA = norm(mA);
  const float alpha = 1/normA;
  debug("computed alpha = %g", alpha);
  for (int i = 0; i < mR->n; ++i) {
    setElem(mR, i, i, alpha);
  }
}

float measureAR(Mat mA, Mat mR) {
  gemm(1, mA, mR, 0, mAR);
  geam(1, mI, -1, mAR, mX);
  return norm(mX);
}

float invert(Mat mA, Mat mR, float maxError, int maxStep) {
  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  computeR0(mA, mR);
  float prevError = 1.0/0.0;
  float error = measureAR(mA, mR);

  int step = 0;
  for (; step < maxStep && error > maxError && error < prevError; ++step) {
    debug("computed approximate inverse R_%d with error measure %g", step,
          error);
    prevError = error;

    invStep(mR);
    error = measureAR(mA, mR);
  }

  clock_gettime(CLOCK_MONOTONIC, &endTime);
  double invTimeS = endTime.tv_sec - startTime.tv_sec +
    (endTime.tv_nsec - startTime.tv_nsec)/1.e9;

  debug("inversion halted at R_%d with error measure %g and took %g seconds",
        step, error, invTimeS);

  if (error >= prevError) {
    warn("R_%d's error measure exceeds that of the preceding inversion step", step);
  } else if (error > maxError) {
    warn("failed to achieve target error measure %g within %d iterations",
         maxError, step);
  }

  return error;
}
