// la.c

#include "invmat.h"

static void computeR0(Mat mA, Mat mR, float(*norm)(Mat)) {
  const float normA = norm(mA);
  const float alpha = 1/normA;
  debug("computed alpha = %g", alpha);
  for (int i = 0; i < mR->n; ++i) {
    setElem(mR, i, i, alpha);
  }
}

float invert(Mat mA, Mat mR, float maxError, int maxStep, bool quadConv) {
  const int n = mA->n;

  debug("initializing work matrices...");
  static Mat mI, mAR, mX;
  if (mA->dev) {
    mI = devNewMat(n); mAR = devNewMat(n); mX = devNewMat(n);
  } else {
    mI = hostNewMat(n); mAR = hostNewMat(n); mX = hostNewMat(n);
  }
  clearMat(mI);
  for (int i = 0; i < n; ++i) {
    setElem(mI, i, i, 1);
  }

  void (*gemm)(float, Mat, Mat, float, Mat) = mA->dev ? cublGemm : blasGemm;
  void (*geam)(float, Mat, float, Mat, Mat) = mA->dev ? cublGeam : blasGeam;
  float (*norm)(Mat)                        = mA->dev ? cublNorm : blasNorm;

  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  computeR0(mA, mR, norm);
  float prevError = 1.0/0.0;
  gemm(1, mA, mR, 0, mAR);
  geam(1, mI, -1, mAR, mX);
  float error = norm(mX);

  int step = 0;
  for (; step < maxStep && error > maxError && error < prevError; ++step) {
    debug("computed approximate inverse R_%d with error measure %g", step,
          error);
    prevError = error;

    if (quadConv) {
      gemm(1, mR, mAR, 0, mX);
      geam(-1, mX, 2, mR, mR);
    } else {
      gemm(1, mR, mAR, 0, mX);
      geam(-3, mX, 3, mR, mR);
      gemm(1, mX, mAR, 1, mR);
    }

    gemm(1, mA, mR, 0, mAR);
    geam(1, mI, -1, mAR, mX);
    error = norm(mX);
  }

  clock_gettime(CLOCK_MONOTONIC, &endTime);
  double invTimeS = endTime.tv_sec - startTime.tv_sec +
    (endTime.tv_nsec - startTime.tv_nsec)/1.e9;

  debug("inversion halted after %g seconds at R_%d with error measure %g",
        invTimeS, step, error);

  if (error >= prevError) {
    warn("R_%d's error measure exceeds that of the preceding inversion step",
         step);
  } else if (error > maxError) {
    warn("failed to achieve target error measure %g within %d iterations",
         maxError, step);
  }

  freeMat(mI); freeMat(mAR); freeMat(mX);
  return error;
}
