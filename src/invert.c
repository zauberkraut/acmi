/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include <time.h>
#include "acmi.h"

/* Quickly computes ||I - AR0|| for R0 = alpha*I. */
static double traceErr(double alpha, Mat mA) {
  return sqrt(MatN(mA) + 1 - 2*alpha*MatTrace(mA));
}

static struct timespec g_startTime;
/* Returns the number of milliseconds elapsed since g_startTime. */
static int msSince() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (time.tv_sec - g_startTime.tv_sec)*1000 +
    (time.tv_nsec - g_startTime.tv_nsec)/1.e6;
}

/* Swaps matrix pointers. */
static void swap(Mat* mp, Mat* np) {
  Mat t = *mp;
  *mp = *np;
  *np = t;
}

/* The inversion algorithm.
   If convRateLimit < 0, |convRateLimit| specifies a multiple of the first
   measured rate to use as the limit. */
double altmanInvert(const Mat mA, Mat* mRp, const int convOrder,
                    const double errLimit, const int msLimit,
                    double convRateLimit, bool safeR0) {
  clock_gettime(CLOCK_MONOTONIC, &g_startTime); // start clock

  debug("initializing work matrices...");
  const int matCount = convOrder < 4 ? 4 : 5;
  Mat mR = MatBuild(mA),
      mAR = MatBuild(mA),
      mX = MatBuild(mA),
      mY = matCount < 5 ? 0 : MatBuild(mA);

  const double alpha = 1/froNorm(mA, false);
  debug("computed alpha = %g", alpha);
  double err = NAN;

  if (safeR0) {
    debug("starting with safe R0");
    transpose(alpha*alpha, mA, mR);
  } else {
    MatClear(mR);
    addId(mR, alpha);
    err = traceErr(alpha, mA);
  }

  for (int iter = 0; msSince() < msLimit; iter++) { // while time remains
    static double prevErr = INFINITY;

    gemm(1, mA, mR, 0, mAR); // mAR <- AR
    if (iter || safeR0) {    // already computed for fast R0
      err = froNorm(mAR, true);
    }
    // rate of convergence
    const double convRate = fabs(err)/pow(fabs(prevErr), convOrder);

    debug("%*sR%d: err = %.4e, μ = %.4e", iter < 10, "", iter, err, convRate);

    if (err <= errLimit) { // we've achieved the desired accuracy
      break;
    }

    // handle divergence
    if (!safeR0 && err >= prevErr && iter < 2) {
      /* Our attempt to exploit the self-adjoint, positive-definite R0 = alpha*I
         failed. Start over using the slow, safe R0 = alpha^2 * A^T instead. */
      debug("diverged, retrying with alternate R0...");
      safeR0 = true;
      transpose(alpha*alpha, mA, mR);
      prevErr = INFINITY;
      iter = -1;
      continue;
    } else if (MatElemSize(mA) < MAX_ELEM_SIZE && convRateLimit > 0 &&
               (convRate >= convRateLimit || err >= prevErr)) {
      debug("diverging, extending to double precision...");
      if (MatDev(mA)) { // quit if we don't have enough GPU memory for promotion
        checkDevMemEnough(MatN(mA), MatElemSize(mA), matCount);
      }
      MatPromote(mA); MatPromote(mR); MatPromote(mAR); MatPromote(mX);
      if (mY) {
        MatPromote(mY);
      }

      double tmp = prevErr;
      prevErr = INFINITY; // our 32-bit error might have been truncated
      if (err >= tmp) {   // if we've fully diverged...
        swap(&mX, &mR);   // ...back up R to its previous value
        iter -= 2;
        continue; // recompute AR
      }
    } else if (err >= prevErr || !isfinite(err)) {
      warn("diverged, R%d is the best we can do", iter - 1);
      swap(&mX, &mR); // back up R to its previous value
      err = prevErr;
      break;          // quit
    } else {
      if (iter == 1 && convRateLimit < 0) { // set rate limit to initial value
        convRateLimit *= -convRate;
      }
      prevErr = err;
    }

    switch (convOrder) { // compute the next iteration
      case 2: // quadratic convergence
        gemm(1, mR, mAR, 0, mX);  // mX  <- RAR
        geam(-1, mX, 2, mR, mAR); // mAR <- 2R - RAR = next R
        swap(&mR, &mAR);          // mR  <- next R, mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R, mAR <- junk
        break;
      case 3: // cubic convergence
        gemm(1, mAR, mAR, 0, mX); // mX  <- (AR)^2
        geam(-3, mAR, 1, mX, mX); // mX  <- -3AR + (AR)^2
        addId(mX, 3);             // mX  <- 3I - 3AR + (AR)^2
        gemm(1, mR, mX, 0, mAR);  // mAR <- R(3I - 3AR + (AR)^2) = next R
        swap(&mR, &mAR);          // mR  <- next R, mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R
        break;
      case 4: // quartic convergence
        gemm(1, mAR, mAR, 0, mX); // mX <- (AR)^2
        geam(-4, mAR, 1, mX, mX); // mX <- -4AR + (AR)^2
        addId(mX, 6);             // mX <- 6I - 4AR + (AR)^2
        gemm(-1, mAR, mX, 0, mY); // mY <- -AR(6I - 4AR + (AR)^2)
        addId(mY, 4);             // mY <- 4I - AR(6I - 4AR + (AR)^2)
        swap(&mR, &mX);           // mX <- previous R
        gemm(1, mX, mY, 0, mR);   // mR <- R(4I - AR(6I - 4AR + (AR)^2))
        break;
      default: fatal("unsupported convergence order: %d", convOrder);
    }
  } // end while

  debug("inversion halted after %d ms", msSince());
  if (err > errLimit) {
    warn("failed to converge to error < %g within %d ms", errLimit, msLimit);
  }

  // cleanup
  MatFree(mAR);
  MatFree(mX);
  if (mY) {
    MatFree(mY);
  }
  *mRp = mR; // return inverted matrix
  return err;
}
