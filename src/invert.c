/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include "acmi.h"

static const double POSDEF_STEP_THRESH = 5;

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

double altmanInvert(Mat mA, Mat* mRp, int convOrder, double errLimit,
                    int msLimit, double convRateLimit) {
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
    const double convRate = fabs(err)/pow(fabs(prevErr), 3);

    debug("%*sR%d: err = %.4e, μ = %.4e", iter < 10, "", iter, err, convRate);

    if (err <= errLimit) {
      break;
    }

    if (posDef && err > prevErr && iter < POSDEF_STEP_THRESH) {
      debug("diverged, retrying with alternate R0...");
      swap(&mX, &mR); // back up R to its previous value
      posDef = false;
      transpose(alpha, mA, mR);
      prevErr = INFINITY;
      iter = -1;
      continue;
    } else if (MatElemSize(mA) < MAX_ELEM_SIZE && convRateLimit > 0 &&
               (convRate > convRateLimit || err > prevErr)) {
      debug("diverging, extending to double precision...");
      MatPromote(mA); MatPromote(mR); MatPromote(mAR); MatPromote(mX);
      prevErr = INFINITY; // our 32-bit error might have been truncated
      iter--;
      continue;
    } else if (err > prevErr) {
      warn("diverged, R%d is the best we can do", iter - 1);
      swap(&mX, &mR); // back up R to its previous value
      err = prevErr;
      break;
    }

    if (iter == 1 && convRateLimit < 0) {
      convRateLimit *= -convRate;
    }
    prevErr = err;

    switch (convOrder) {
      case 2:
        gemm(1, mR, mAR, 0, mX);  // mX  <- RAR
        geam(-1, mX, 2, mR, mAR); // mAR <- 2R - RAR = next R
        swap(&mR, &mAR);          // mR  <- next R,     mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R, mAR <- junk
        break;
      case 3:
        gemm(1, mAR, mAR, 0, mX); // mX  <- (AR)^2
        add3I(mX);                // mX  <- 3I + (AR)^2
        geam(-3, mAR, 1, mX, mX); // mX  <- 3I - 3AR + (AR)^2
        gemm(1, mR, mX, 0, mAR);  // mAR <- R(3I - 3AR + (AR)^2) = next R
        swap(&mR, &mAR);          // mR  <- next R,     mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R, mAR <- junk
        break;
      default: fatal("unsupported convergence order: %d", convOrder);
    }
  }

  debug("inversion halted after %d ms", msSince());
  if (err > errLimit) {
    warn("failed to converge to error < %g within %d ms", errLimit, msLimit);
  }

  MatFree(mAR); MatFree(mX);
  *mRp = mR;
  return err;
}
