// main.c

#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include "invmat.h"

const int MAX_PATH_LEN = 255;
const Impl DEFAULT_IMPL = CUBLAS_IMPL;
const float DEFAULT_MAX_ERROR = 0.0001;
const int DEFAULT_MAX_STEP = 20;

extern char *optarg;
extern int optind, opterr, optopt;

void usage() {
  fatal("GPU-Parallelized Matrix Inverter, J. Treadwell, 2016\n\n"
        "Usage:\n  invmat [options] <input matrix path>\n\n"
        "Options:\n"
        "  -h          These instructions\n"
        "  -q          Disable logging\n"
        "  -i          Print matrix file info and exit\n"
        "  -t          Test mode: don't output the computed inverse\n"
        "  -2          Employ quadratic instead of cubic convergence\n"
        "  -m <mode>   Select implementation to run (default: cublas)\n"
        "  -p <N>      Use N-bit floating-point matrix elements\n"
        "  -e <real>   Set max inversion error (default: %g)\n"
        "  -n <count>  Set max iteration to compute (default: %d)\n"
        "  -r <N>      Invert a random NxN integer matrix\n"
        "  -R <N>      Invert a random NxN real matrix\n"
        "  -o <path>   Set output path for uninverted random matrix\n"
        "  -s <hex>    Set PRNG seed\n\n"
        "MatrixMarket files are accepted as input.\n"
        "Computed inverses are written in MatrixMarket format to stdout.\n\n"
        "Modes available through -m option:\n"
        "  blas    Pure software BLAS implementation; no Nvidia GPU needed\n"
        "  cublas  Nvidia CUDA BLAS implementation\n"
        "  lu      Nvidia LU-factorization inversion without iteration\n\n"
        "Use the -p option to set floating-point precision to 16, 32 or 64-bits.",
        DEFAULT_MAX_ERROR, DEFAULT_MAX_STEP);
}

int main(const int argc, char* const argv[]) {
  bool quadConv = false;
  bool infoMode = false;
  bool testMode = false;
  Impl impl = DEFAULT_IMPL;
  float maxError = DEFAULT_MAX_ERROR;
  int maxStep = DEFAULT_MAX_STEP;
  int randDim = 0;
  bool randRealMode = false;
  char* randOutPath = 0;
  unsigned prngSeed = 0;

  if (argc == 1) {
    usage();
  }

  opterr = 0;
  int opt;

  while ((opt = getopt(argc, argv, "hqit2m:e:n:r:R:o:s:")) != -1) {
    switch (opt) {
      case 'h': usage();
      case 'q': setVerbose(false); break;
      case 'i': infoMode = true; break;
      case 't': testMode = true; break;
      case '2': quadConv = true; break;

      case 'm': {
        if (!strncmp(optarg, "blas", 8)) {
          impl = BLAS_IMPL;
        } else if (!strncmp(optarg, "cublas", 8)) {
          impl = CUBLAS_IMPL;
        } else if (!strncmp(optarg, "lu", 8)) {
          impl = LU_IMPL;
        } else {
          fatal("supported modes are: \"blas\", \"cublas\", \"lu\"");
        }
        break;
      }
      case 'e': {
        char* parsePtr;
        maxError = strtof(optarg, &parsePtr);
        if (parsePtr - optarg != strlen(optarg) || maxError < 0 ||
            maxError >= 1 || errno == ERANGE) {
          fatal("max error measure must be a real on [0, 1)");
        }
        break;
      }
      case 'n': {
        char* parsePtr;
        maxStep = strtol(optarg, &parsePtr, 10);
        if (parsePtr - optarg != strlen(optarg) || maxStep < 0 ||
            maxStep > 1000 || errno == ERANGE) {
          fatal("invalid step limit");
        }
        break;
      }
      case 'R':
        randRealMode = true;
      case 'r': {
        char* parsePtr;
        randDim = strtol(optarg, &parsePtr, 10);
        if (parsePtr - optarg != strlen(optarg) || randDim <= 1 ||
            randDim > MAX_MAT_DIM || errno == ERANGE) {
          fatal("invalid random matrix dimension");
        }
        break;
      }
      case 'o': {
        randOutPath = strndup(optarg, MAX_PATH_LEN);
        break;
      }
      case 's': {
        char* parsePtr;
        prngSeed = strtol(optarg, &parsePtr, 16);
        if (parsePtr - optarg != strlen(optarg) || prngSeed < 1 ||
            errno == ERANGE) {
          fatal("invalid 32-bit hexadecimal seed");
        }
        break;
      }

      case '?': {
        switch (optopt) {
          case 'm': case 'e': case 'n': case 'r': case 'R': case 'o':
          case 's': {
            fatal("option -%c missing argument", optopt);
            continue;
          }
        }
      }
      default: {
        char cbuf[21];
        if (isprint(optopt)) {
          snprintf(cbuf, sizeof(cbuf), "%c", optopt);
        } else {
          snprintf(cbuf, sizeof(cbuf), "<0x%x>", optopt);
        }
        fatal("invalid option: -%s", cbuf);
      }
    }
  }

  Mat mA = 0;

  if (randDim) {
    if (optind < argc) {
      fatal("an input file or unexpected argument was given for random matrix "
            "mode");
    }
    if (infoMode) fatal("info mode doesn't apply to randomly-generated "
                        "matrices");

    mA = impl == BLAS_IMPL ? hostNewMat(randDim) : devNewMat(randDim);

    if (!prngSeed) prngSeed = time(0);
    debug("seeding PRNG with %x", prngSeed);
    srand(prngSeed);

    debug("generating random %dx%d %s matrix...", randDim, randDim,
          randRealMode ? "real" : "integer");
    if (randRealMode) genRandRealMat(mA, randOutPath);
    else              genRandIntMat(mA, randOutPath);

    if (randOutPath) free(randOutPath);
    debug("inverting random matrix...");
  } else {
    if (optind >= argc) {
      fatal("missing input matrix filename");
    }
    if (optind < argc - 1) {
      fatal("unexpected argument: %s", argv[optind+1]);
    }
    if (randOutPath || prngSeed) {
      fatal("options -o, -s only apply to random mode");
    }

    const char* matPath = argv[optind];

    MatInfo info = readMatInfo(matPath);
    if (infoMode) {
      return 0;
    }

    mA = impl == BLAS_IMPL ? hostNewMat(info.n) : devNewMat(info.n);
    loadMat(mA, info);
    debug("inverting %s ...", matPath);
  }

  Mat mR = mA->dev ? devNewMat(mA->n) : hostNewMat(mA->n);
  clearMat(mR);

  switch (impl) {
    case CUBLAS_IMPL:
      debug("inverting using cuBLAS");
      cublInit(mA->n);
    case BLAS_IMPL:
      if (quadConv) debug("inverting quadratically");
      invert(mA, mR, maxError, maxStep, quadConv);
      break;
    case LU_IMPL:
      cublInit(mA->n);
      debug("inverting using cuBLAS LU-factorization");
      luInvert(mA, mR);
      break;
  }

  if (!testMode && mR) writeMat(stdout, mR);

  freeMat(mA);
  if (mR) freeMat(mR);

  switch (impl) {
    case CUBLAS_IMPL:
    case LU_IMPL:
      cublShutDown();
      break;
    default:
      break;
  }

  return 0;
}
