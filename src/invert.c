/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include "acmi.h"

double altmanInvert(Mat mA, Mat mR, double maxError, int maxStep, bool quadConv)
{
  debug("initializing work matrices...");
  Mat mI = MatBuild(mA), mAR = MatBuild(mA), mX = MatBuild(mA);
  MatClear(mI);
  for (int i = 0; i < MatN(mA); ++i) {
    MatPut(mI, i, i, 1);
  }

  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  const double normA = norm(mA);
  const double alpha = 1/normA;

  MatClear(mR);
  for (int i = 0; i < MatN(mR); ++i) {
    MatPut(mR, i, i, alpha);
  }

  debug("computed alpha = %g", alpha);
  double error, prevError = INFINITY;

  gemm(1, mA, mR, 0, mAR);
  error = sqrt(MatN(mA) + 1 - 2*alpha*MatTrace(mA));

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

  MatFree(mI); MatFree(mAR); MatFree(mX);
  return error;
}
