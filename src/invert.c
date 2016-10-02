/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include "acmi.h"

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

double altmanInvert(const Mat mA, Mat* mRp, const int convOrder,
                    const double errLimit, const int msLimit,
                    double convRateLimit, bool safeR0) {
  clock_gettime(CLOCK_MONOTONIC, &g_startTime);

  debug("initializing work matrices...");
  Mat mR = MatBuild(mA),
      mAR = MatBuild(mA),
      mX = MatBuild(mA),
      mY = convOrder > 3 ? MatBuild(mA) : 0;

  const double alpha = 1/norm(mA);
  debug("computed alpha = %g", alpha);
  double err = NAN;

  if (safeR0) {
    debug("starting with safe R0");
    transpose(alpha*alpha, mA, mR);
  } else {
    MatClear(mR);
    setDiag(mR, alpha);
    err = traceErr(alpha, mA);
  }

  for (int iter = 0; msSince() < msLimit; iter++) {
    static double prevErr = INFINITY;

    gemm(1, mA, mR, 0, mAR); // mAR <- AR
    if (iter || safeR0) {
      err = normSubFromI(mAR);
    }
    // rate of convergence
    const double convRate = fabs(err)/pow(fabs(prevErr), convOrder);

    debug("%*sR%d: err = %.4e, μ = %.4e", iter < 10, "", iter, err, convRate);

    if (err <= errLimit) { // we've achieved the desired accuracy
      break;
    }

    // handle divergence
    if (!safeR0 && err >= prevErr && iter < 2) {
      /* Our attempt to exploit the self-adjoint R0 = alpha*I failed. Use the
         slower R0 = alpha*A*A^T starting point instead. */
      debug("diverged, retrying with alternate R0...");
      safeR0 = true;
      transpose(alpha*alpha, mA, mR);
      prevErr = INFINITY;
      iter = -1;
      continue;
    } else if (MatElemSize(mA) < MAX_ELEM_SIZE && convRateLimit > 0 &&
               (convRate >= convRateLimit || err >= prevErr)) {
      debug("diverging, extending to double precision...");
      MatPromote(mA); MatPromote(mR); MatPromote(mAR); MatPromote(mX);
      if (mY) {
        MatPromote(mY);
      }
      if (false) { // TODO
        swap(&mX, &mR); // back up R to its previous value
        iter--;
      }
      prevErr = INFINITY; // our 32-bit error might have been truncated
      //iter--; // TODO
      //continue;
    } else if (err >= prevErr || !isfinite(err)) {
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
        swap(&mR, &mAR);          // mR  <- next R, mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R, mAR <- junk
        break;
      case 3:
        gemm(1, mAR, mAR, 0, mX); // mX  <- (AR)^2
        geam(-3, mAR, 1, mX, mX); // mX  <- -3AR + (AR)^2
        addDiag(mX, 3);           // mX  <- 3I - 3AR + (AR)^2
        gemm(1, mR, mX, 0, mAR);  // mAR <- R(3I - 3AR + (AR)^2) = next R
        swap(&mR, &mAR);          // mR  <- next R, mAR <- previous R
        swap(&mX, &mAR);          // mX  <- previous R
        break;
      case 4:
        gemm(1, mAR, mAR, 0, mX); // mX <- (AR)^2
        geam(-4, mAR, 1, mX, mX); // mX <- -4AR + (AR)^2
        addDiag(mX, 6);           // mX <- 6I - 4AR + (AR)^2
        gemm(-1, mAR, mX, 0, mY); // mY <- -AR(6I - 4AR + (AR)^2)
        addDiag(mY, 4);           // mY <- 4I - AR(6I - 4AR + (AR)^2)
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

  MatFree(mAR);
  MatFree(mX);
  if (mY) {
    MatFree(mY);
  }
  *mRp = mR;
  return err;
}
