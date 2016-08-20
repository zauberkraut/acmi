/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include "acmi.h"

const double POSDEF_STEP_THRESH = 5;

static double traceError(double alpha, Mat mA) {
  return sqrt(MatN(mA) + 1 - 2*alpha*MatTrace(mA));
}

static void swap(Mat* mp, Mat* np) {
  Mat t = *mp;
  *mp = *np;
  *np = t;
}

double altmanInvert(Mat mA, Mat* mRp, double maxError, int maxStep,
                    bool quadConv) {
  debug("initializing work matrices...");
  Mat mAR = MatBuild(mA), mX = MatBuild(mA);

  struct timespec startTime, endTime;
  clock_gettime(CLOCK_MONOTONIC, &startTime);

  const double normA = norm(mA);
  const double alpha = 1/normA;
  debug("computed alpha = %g", alpha);

  bool posDef = true;
  Mat mR = MatBuild(mA);
  MatClear(mR);
  for (int i = 0; i < MatN(mR); ++i) {
    MatPut(mR, i, i, alpha);
  }

  gemm(1, mA, mR, 0, mAR);

  double error, prevError = INFINITY;
  error = traceError(alpha, mA);

  int step = 0;
  for (; step < maxStep && error > maxError; ++step) {
    debug("computed approximate inverse R_%d with error measure %g", step,
          error);

    if (error > prevError) {
      debug("R_%d diverged", step);
      // back up R to its previous value
      swap(&mX, &mR);
      step--;

      if (step < POSDEF_STEP_THRESH && posDef) {
        debug("retrying with non-positive-definite R0...");
        posDef = false;
        transpose(alpha, mA, mR);
        gemm(1, mA, mR, 0, mAR);
        prevError = INFINITY;
        // replace with sweep squares
        error = normSubFromI(mAR);
        step = -1;
        continue;
      } else if (!MatDouble(mA)) {
        debug("retrying with double-precision...");
        MatWiden(mA); MatWiden(mR); MatWiden(mAR); MatWiden(mX);
      } else {
        warn("R_%d is the best we can do", step);
        break;
      }
    }

    prevError = error;

    if (quadConv) {
      gemm(1, mR, mAR, 0, mX);
      geam(-1, mX, 2, mR, mR);
    } else {
      gemm(1, mAR, mAR, 0, mX);
      // add 3I
      add3I(mX);
      // minus 3 mat
      geam(-3, mAR, 1, mX, mX);
      // new R (overwriting AR, since we'll recompute that anyway)
      gemm(1, mR, mX, 0, mAR);
      // put the new R where it belongs; AR now has the old R
      swap(&mR, &mAR);
    }

    // new AR
    gemm(1, mA, mR, 0, mX);
    // compute error
    error = normSubFromI(mX);
    // swap mX and mAR so the former has the old R and the latter the new AR
    swap(&mX, &mAR);
  }

  clock_gettime(CLOCK_MONOTONIC, &endTime);
  double invTimeS = endTime.tv_sec - startTime.tv_sec +
    (endTime.tv_nsec - startTime.tv_nsec)/1.e9;

  debug("inversion halted after %g seconds at R_%d with error measure %g",
        invTimeS, step, error);

  if (error > maxError) {
    warn("failed to achieve target error measure %g within %d iterations",
         maxError, step);
  }

  MatFree(mAR); MatFree(mX);
  *mRp = mR;
  return error;
}
