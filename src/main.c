// main.c

#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include "invmat.h"

const float DEF_MAX_ERROR = 0.0001;
const int DEF_MAX_STEP = 20;
const int MAX_PATH_LEN = 255;

extern char *optarg;
extern int optind, opterr, optopt;

void usage() {
  fatal("GPU-Parallelized Matrix Inverter, J. Treadwell, 2016\n"
        "Usage:\n  invmat [options] <input matrix path>\n\n"
        "Options:\n"
        "  -h                   These instructions\n"
        "  -q                   Disable logging\n"
        "  -i                   Print matrix file info and exit\n"
        "  -t                   Test mode: don't output the computed inverse\n"
        "  -2                   Employ quadratic instead of cubic convergence\n"
        "  -m <cpu|cublas|cuda> Select implementation to run (default: cpu)\n"
        "  -e <real>            Set max inversion error (default: %g)\n"
        "  -n <count>           Set max iteration to compute (default: %d)\n"
        "  -r <N> [-o <path>]   Invert a random NxN integer matrix\n"
        "  -R <N> [-o <path>]   Invert a random NxN real matrix\n"
        "  -s <32-bit hex>      Specify PRNG seed\n\n"
        "MatrixMarket files are accepted as input.\n"
        "Computed inverses are written in MatrixMarket format to stdout.\n"
        "Random modes take an optional path to which to write the original,\n"
        "randomly-generated matrix and don't take an input file.",
        DEF_MAX_ERROR, DEF_MAX_STEP);
}

int main(const int argc, char* const argv[]) {
  bool quadConv = false;
  bool infoMode = false;
  bool testMode = false;
  Impl impl = CPU_IMPL;
  float maxError = DEF_MAX_ERROR;
  int maxStep = DEF_MAX_STEP;
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
        if (!strncmp(optarg, "cublas", 8)) {
          impl = CUBLAS_IMPL;
        } else if (!strncmp(optarg, "cuda", 8)) {
          impl = CUDA_IMPL;
        } else if (!strncmp(optarg, "lu", 8)) {
          impl = LU_IMPL;
        } else if (strncmp(optarg, "cpu", 8)) {
          fatal("implementation must be one of \"cpu\", \"cublas\" or \"cuda\"");
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

    if (!prngSeed) prngSeed = time(0);
    debug("seeding PRNG with %x", prngSeed);
    srand(prngSeed);

    init(randDim, impl, quadConv);

    debug("generating random %dx%d %s matrix...", randDim, randDim,
          randRealMode ? "real" : "integer");
    mA = randRealMode ? randRealMat(randDim) : randIntMat(randDim);

    if (randOutPath) {
      FILE* frand = fopen(randOutPath, "w");
      if (!frand) fatal("couldn't open %s to write random matrix");
      writeMat(frand, mA);
      fclose(frand);
      free(randOutPath);
    }
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

    init(info.n, impl, quadConv);

    mA = loadMat(info);
    debug("inverting %s ...", matPath);
  }

  Mat mR = zeroMat(mA->n);

  if (impl == LU_IMPL) {
    luInvert(mA, mR);
  } else {
    invert(mA, mR, maxError, maxStep);
  }

  if (!testMode && mR) writeMat(stdout, mR);

  freeMat(mA);
  if (mR) freeMat(mR);
  shutDown();
  return 0;
}
