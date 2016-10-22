/* invert.c

   ACMI convergent inversion algorithm implementation. */

#include <time.h>
#include "acmi.h"

enum { MAX_RESTART_ITER = 2 };

/* Quickly computes ||I - AR0|| for R0 = alpha*I. */
static double traceErr(double alpha, Mat mA) {
  return sqrt(MatN(mA) + 1 - 2*alpha*trace(mA));
}

/* Returns the number of milliseconds elapsed since start. */
static int msSince(struct timespec* start) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (time.tv_sec - start->tv_sec)*1000 +
    (time.tv_nsec - start->tv_nsec)/1.e6;
}

/* Swaps matrix pointers. */
static void swap(Mat* mp, Mat* np) {
  Mat t = *mp;
  *mp = *np;
  *np = t;
}

/* The inversion algorithm.

   If convRateLimit < 0, |convRateLimit| specifies a multiple of the
   first measured rate to use as the limit. */
double altmanInvert(const Mat mA, Mat* const mRp,
                    const int convOrder, const double errLimit,
                    const int msLimit, double convRateLimit,
                    bool safeR0) {
  static struct timespec start;

  debug("initializing work matrices...");
  clock_gettime(CLOCK_MONOTONIC, &start); // start clock
  const int matCount = convOrder < 4 ? 4 : 5;
  Mat mR = MatBuild(mA), // allocate matrices with A's dimensions
      mAR = MatBuild(mA),
      mX = MatBuild(mA),
      mY = matCount < 5 ? 0 : MatBuild(mA);

  const double alpha = 1/nrm2(mA);
  debug("computed alpha = %g", alpha);
  double err = NAN;

  if (safeR0) {
    debug("starting with safe R0");
    transpose(alpha*alpha, mA, mR);
  } else {
    MatClear(mR);
    addId(alpha, mR);
    err = traceErr(alpha, mA);
  }

  int iter, ms;
  for (iter = 0; (ms = msSince(&start)) < msLimit || !msLimit;
       iter++) { // while time remains
    static double prevErr = INFINITY;
    static int prevMS = 0;

    gemm(1, mA, mR, 0, mAR); // mAR <- AR
    if (iter || safeR0) {    // don't overwrite above optimization
      err = minusIdNrm2(mAR);
    }
    // rate of convergence
    const double convRate = fabs(err)/pow(fabs(prevErr), convOrder);

    debug("%*sR%d: err = %.4e, μ = %.4e  (%.3fs)", iter < 10, "",
          iter, err, convRate, (ms - prevMS)/1000.);
    prevMS = ms;

    if (err <= errLimit) { // we've achieved the desired accuracy
      break;
    }

    // handle divergence
    if (!safeR0 && err >= prevErr && iter <= MAX_RESTART_ITER) {
      /* Our attempt to exploit the self-adjoint, positive-definite
         R0 = alpha*I failed. Start over using the slow, safe
         R0 = alpha^2 * A^T instead. */
      debug("diverged, retrying with alternate R0...");

      safeR0 = true;
      transpose(alpha*alpha, mA, mR);
      prevErr = INFINITY;
      iter = -1;

      continue;
    } else if (MatElemSize(mA) < MAX_ELEM_SIZE && convRateLimit > 0
               && (convRate >= convRateLimit || err >= prevErr)) {
      debug("diverging, extending to double precision...");

      if (MatDev(mA) && // quit if we lack GPU memory for promotion
          !checkDevMemEnough(MatN(mA), MatElemSize(mA), matCount)) {
        debug("not enough device RAM; halting");
        if (err >= prevErr) { // back up if last iter were better
          swap(&mX, &mR);
          iter--;
        }
        break;
      }

      MatPromote(mA); MatPromote(mR); MatPromote(mAR);
      MatPromote(mX);
      if (mY) {
        MatPromote(mY);
      }

      double tmp = prevErr;
      prevErr = INFINITY; // in case error shall reinflate
      if (err >= tmp) {   // if we've fully diverged...
        debug("backing up to reduce error");
        swap(&mX, &mR);
        iter -= 2;
        continue; // recompute AR
      }
    } else if (err >= prevErr || !isfinite(err)) {
      warn("diverged, R%d is the best we can do", iter - 1);

      swap(&mX, &mR); // back up R to its previous value
      err = prevErr;
      break;          // quit
    } else {
      if (iter == 1 && convRateLimit < 0) {
        // set rate limit to initial value
        convRateLimit *= -convRate;
      }

      prevErr = err;
    }

    switch (convOrder) { // compute the next iteration
      case 2: // quadratic convergence
        gemm(1, mR, mAR, 0, mX);  // mX  <- RAR
        geam(-1, mX, 2, mR, mAR); // mAR <- 2R - RAR = next R
        swap(&mR, &mAR);          // mR  <- next R, mAR <- prev R
        swap(&mX, &mAR);          // mX  <- previous R, mAR <- junk
        break;
      case 3: // cubic convergence
        gemm(1, mAR, mAR, 0, mX); // mX  <- (AR)^2
        geam(-3, mAR, 1, mX, mX); // mX  <- -3AR + (AR)^2
        addId(3, mX);             // mX  <- 3I - 3AR + (AR)^2
        gemm(1, mR, mX, 0, mAR);  // mAR <- R(3I - 3AR + (AR)^2)
        swap(&mR, &mAR);          // mR  <- next R, mAR <- prev R
        swap(&mX, &mAR);          // mX  <- previous R
        break;
      case 4: // quartic convergence
        gemm(1, mAR, mAR, 0, mX); // mX <- (AR)^2
        geam(-4, mAR, 1, mX, mX); // mX <- -4AR + (AR)^2
        addId(6, mX);             // mX <- 6I - 4AR + (AR)^2
        gemm(-1, mAR, mX, 0, mY); // mY <- -AR(6I - 4AR + (AR)^2)
        addId(4, mY);             // mY <- 4I - AR(6I-4AR+(AR)^2)
        swap(&mR, &mX);           // mX <- previous R
        gemm(1, mX, mY, 0, mR);   // mR <- R(4I - AR(6I-4AR+(AR)^2))
        break;
      default:
        fatal("unsupported convergence order: %d", convOrder);
    }
  } // end while

  ms = msSince(&start);
  int minutes = ms/60000;
  double seconds = (ms - minutes*60000)/1000.;
  debug("inversion halted in %d iterations after %dm%.3fs", iter,
        minutes, seconds);
  if (err > errLimit) {
    warn("failed to converge to error < %g within %d ms", errLimit,
         msLimit);
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
