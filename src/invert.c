/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include "acmi.h"

const double POSDEF_STEP_THRESH = 5;
const double MAX_CONV_RATE_FACTOR = 10;

static double traceErr(double alpha, Mat mA) {
  return sqrt(MatN(mA) + 1 - 2*alpha*MatTrace(mA));
}

static struct timespec g_startTime;
static int msSince() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (time.tv_sec - g_startTime.tv_sec)*1000 +
    (time.tv_nsec - g_startTime.tv_nsec)/1.e6;
}

static void swap(Mat* mp, Mat* np) {
  Mat t = *mp;
  *mp = *np;
  *np = t;
}

double altmanInvert(Mat mA, Mat* mRp, double errLimit, int msLimit,
                    bool quadConv) {
  clock_gettime(CLOCK_MONOTONIC, &g_startTime);

  debug("initializing work matrices...");
  Mat mAR = MatBuild(mA), mX = MatBuild(mA);

  const double alpha = 1/norm(mA);
  debug("computed alpha = %g", alpha);

  Mat mR = MatBuild(mA);
  MatClear(mR);
  setDiag(mR, alpha);
  bool posDef = true;
  double err = traceErr(alpha, mA);

  for (int iter = 0; msSince() < msLimit; iter++) {
    gemm(1, mA, mR, 0, mAR);
    // TODO: replace with sweep squares
    if (iter || !posDef) {
      err = normSubFromI(mAR);
    }
    static double prevErr = INFINITY;
    // rate of convergence
    const double convRate = err/pow(prevErr, 3);

    debug("%sR%d: err=%11g, μ=%11g", iter < 10 ? " " : "", iter, err, convRate);
    if (err <= errLimit) {
      break;
    }

    if (posDef && err > prevErr && iter < POSDEF_STEP_THRESH) {
      debug("diverged, retrying with non-positive-definite R0...");
      swap(&mX, &mR); // back up R to its previous value
      posDef = false;
      transpose(alpha, mA, mR);
      prevErr = INFINITY;
      iter = -1;
      continue;
    } else if (!MatDouble(mA) && convRate > 1) {
      debug("diverging, extending to double precision...");
      // TODO: MatExtend
      MatWiden(mA); MatWiden(mR); MatWiden(mAR); MatWiden(mX);
      prevErr = INFINITY;   // our 32-bit error might have been truncated
      iter--;
      continue;
    } else if (err > prevErr) {
      warn("diverged, R%d is the best we can do", iter - 1);
      swap(&mX, &mR); // back up R to its previous value
      err = prevErr;
      break;
    }

    prevErr = err;

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
      // put old R in X, junk in AR
      swap(&mX, &mAR);
    }
  }

  debug("inversion halted after %g seconds", msSince()/1000.);
  if (err > errLimit) {
    warn("failed to converge to error < %g within %g seconds",
         errLimit, msLimit/1000.);
  }

  MatFree(mAR); MatFree(mX);
  *mRp = mR;
  return err;
}
