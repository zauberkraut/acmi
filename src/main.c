/* main.c

   ACMI entry and setup. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>
#include "acmi.h"

enum {
  MAX_PATH_LEN = 255,
  MIN_ELEM_BITS = 32,
  MAX_ELEM_BITS = 64,
  DEFAULT_ELEM_SIZE = 4,
  MIN_CONV_ORDER = 1,
  MAX_CONV_ORDER = 4,
  DEFAULT_CONV_ORDER = 3,
  DEFAULT_MS_LIMIT = 60000, // one minute
  MAX_MS_LIMIT = 86400000   // one day
};

static const double
  DEFAULT_ERR_LIMIT = 1e-5,
  #define MIN_CONV_RATE 1e-5
  MIN_CONV_RATE_FACTOR = 1. + MIN_CONV_RATE,
  MAX_CONV_RATE = 1e6,
  DEFAULT_CONV_RATE_LIMIT = -1.01;

static const char* DEFAULT_CONV_ORDER_STR = "cubic";
static const char* DEFAULT_CONV_RATE_LIMIT_STR = "x1.01";

// for dealing with getopt arguments
extern char *optarg;
extern int optind, opterr, optopt;

/* Parses an integer argument of the given radix from the command line, aborting
   after printing errMsg if an error occurs or the integer exceeds the given
   bounds. */
long parseInt(int radix, long min, long max, const char* errMsg) {
  char* parsePtr = 0;
  long l = strtol(optarg, &parsePtr, radix);
  if (parsePtr - optarg != strlen(optarg) || l < min || l > max ||
      ERANGE == errno) {
    fatal(errMsg);
  }
  return l;
}

double parseFloat(double min, double maxEx, const char* errMsg) {
  char* parsePtr;
  double v = strtof(optarg, &parsePtr);
  if (parsePtr - optarg != strlen(optarg) || v < min || v >= maxEx ||
      ERANGE == errno) {
    fatal(errMsg);
  }
  return v;
}

/* Aborts if the given path can't be written to. */
void checkWriteAccess(const char* path) {
  const int len = strlen(path);
  if (len > MAX_PATH_LEN) {
    fatal("input file path exceeds %d characters", MAX_PATH_LEN);
  }

  // reject if path is a directory
  DIR* dir = 0;
  if ('/' == path[len-1] || '\\' == path[len-1] || (dir = opendir(path))) {
    if (dir) {
      closedir(dir);
    }
    fatal("%s is a directory", path);
  }

  if (access(path, F_OK)) { // if file does not exist
    // check if we can create it (i.e. can we write to the directory?)
    char* pathCopy = strndup(path, MAX_PATH_LEN);
    const char* dir = dirname(pathCopy);
    bool dirNotWritable = access(dir, W_OK | X_OK);
    free(pathCopy);
    if (dirNotWritable) {
      fatal("can't write to directory %s", dir);
    }
  } else if (access(path, W_OK)) { // file exists; may we overwrite it?
    fatal("can't write to %s", path);
  }
}

void usage() {
  printf("ACMI Convergent Matrix Inverter\nJ. Treadwell, 2016\n\n"
         "Usage:\n  acmi [options] <input file>\n\n"
         "  Currently, only Matrix Market files are supported as input and output.\n"
         "  To generate and invert a randomly matrix, enter the matrix dimension\n"
         "  in lieu of the input file, prepending with '?', '$' or 'b' to generate an\n"
         "  integer, diagonally-dominant or boolean matrix, respectively.\n\n"
         "Options:\n"
         "  -h          These instructions\n"
         "  -q          Disable logging\n"
         "  -i          Print matrix file info and exit\n"
         "  -c          Perform all computations in software without the GPU\n"
         "  -o <path>   Output computed matrix inverse to path\n"
         "  -r <order>  Set the order of convergence (1-4, default: %s)\n"
         "  -p <#bits>  Set initial matrix element floating-point precision\n"
         "              (32 or 64, default: %d)\n"
         "  -e <+real>  Set inversion error limit (default: %g)\n"
         "  -t <ms>     Set inversion time limit in ms (default: %d ms)\n"
         "  -m <+real>  Set max convergence rate allowed using single-precision; prepend\n"
         "              with 'x' to use a multiple (>= 1) of the starting rate\n"
         "              (default: %s)\n"
         "Random matrix options:\n"
         "  -s          Generate symmetric matrix\n"
         "  -O <path>   Output generated, uninverted random matrix to path\n"
         "  -x <hex>    Set PRNG seed (not yet portable)\n\n",
         DEFAULT_CONV_ORDER_STR, 8*DEFAULT_ELEM_SIZE, DEFAULT_ERR_LIMIT, DEFAULT_MS_LIMIT,
         DEFAULT_CONV_RATE_LIMIT_STR);
  exit(0);
}

int main(int argc, char* argv[]) {
  bool infoMode = false;
  char* outPath = 0;
  bool softMode = false;
  int elemSize = DEFAULT_ELEM_SIZE;
  int convOrder = DEFAULT_CONV_ORDER;
  double errLimit = DEFAULT_ERR_LIMIT;
  int msLimit = DEFAULT_MS_LIMIT;
  double convRateLimit = DEFAULT_CONV_RATE_LIMIT;
  int randDim = 0;
  bool randSymm = false;
  char* randOutPath = 0;
  unsigned prngSeed = 0;

  if (1 == argc) {
    usage();
  }

  opterr = 0;
  int opt;
  while ((opt = getopt(argc, argv, "hqico:r:p:e:t:m:sO:x:")) != -1) {
    switch (opt) {
      int i;
      double d;

    case 'h': usage();           break;
    case 'q': setVerbose(false); break;
    case 'i': infoMode = true;   break;
    case 'c': softMode = true;   break;
    case 's': randSymm = true;   break;

    case 'o':
      checkWriteAccess(optarg);
      outPath = strndup(optarg, MAX_PATH_LEN);
      break;
    case 'p':
      i = (int)parseInt(10, MIN_ELEM_BITS, MAX_ELEM_BITS,
                        "invalid floating-point precision");
      d = log2(i);
      if (d != floor(d)) {
        fatal("invalid floating-point precision");
      }
      elemSize = i/8;
      break;
    case 'r':
      convOrder = (int)parseInt(10, MIN_CONV_ORDER, MAX_CONV_ORDER,
                                "conversion order must be 1-4");
      break;
    case 'e':
      errLimit = parseFloat(0, 1,
                            "error limit must be a real on [0, 1)");
      break;
    case 't':
      msLimit = (int)parseInt(10, 0, MAX_MS_LIMIT, "invalid time limit in ms");
      break;
    case 'm':
      i = 1;
      d = MIN_CONV_RATE;
      if (*optarg == 'x') {
        optarg++;
        i = -1;
        d = MIN_CONV_RATE_FACTOR;
      }
      convRateLimit = i*parseFloat(d, MAX_CONV_RATE,
                                   "invalid convergence rate limit");
      break;
    case 'O':
      checkWriteAccess(optarg);
      randOutPath = strndup(optarg, MAX_PATH_LEN);
      break;
    case 'x':
      prngSeed = (unsigned)parseInt(16, 1, UINT_MAX,
                                    "invalid 32-bit hexadecimal seed");
      break;

    case '?':
      switch (optopt) {
      case 'o': case 'm': case 'e': case 'n': case 'O': case 'x':
        fatal("option -%c missing argument", optopt);
      }
    default: { // report invalid option character
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

  optarg = argv[optind];
  // enforce exactly one non-option parameter describing the matrix to invert
  if (!optarg || 0 == strlen(optarg)) {
    fatal("missing input file");
  }
  if (optind < argc - 1) {
    fatal("unexpected argument: %s", argv[optind+1]);
  }
  if ('?' == optarg[0]) { // random mode
    optarg++; // parse remainder of argument as the matrix dimension
    randDim = (int)parseInt(10, 2, MAX_MAT_DIM,
                            "invalid random matrix dimension");
  } else {
    if (randSymm || randOutPath || prngSeed) {
      fatal("options -r, -s, -O and -x apply only to random matrices");
    }
  }

  if (elemSize == 2 && softMode) {
    fatal("half-precision floating-point only supported on the GPU");
  }

  Mat mA = 0;

  if (!softMode) {
    debug("%g MiB device memory available", mibibytes(cuMemAvail()));
  }
  const int loadElemSize = elemSize == 2 ? 4 : elemSize;

  if (randDim) { // random mode
    if (!prngSeed) prngSeed = time(0); // if user supplied no seed, use time
    debug("seeding PRNG with %x", prngSeed);
    srand(prngSeed);

    debug("generating %d-bit random %dx%d%s matrix...", 8*loadElemSize,
          randDim, randDim, randSymm ? " symmetric" : "");
    mA = MatRandDiagDom(randDim, loadElemSize, randSymm);

    if (randOutPath) { // optionally write randomly-generated matrix
      MatWrite(mA, randOutPath);
      free(randOutPath);
    }
  } else { // load matrix from given file
    debug("loading %s", optarg);
    mA = MatLoad(optarg, loadElemSize, infoMode);
  }

  if (elemSize == 2) {
    debug("demoting loaded matrix to 16-bit, original size: %g MiB",
          mibibytes(MatSize(mA)));
    MatDemote(mA);
  }

  const double matMiB = mibibytes(MatSize(mA));
  debug("%g MiB/matrix; allocating %g MiB total", matMiB, 4*matMiB);

  if (infoMode) {
    debug("matrix info-only mode: terminating");
    exit(0);
  }

  if (!softMode) { // upload source matrix to GPU
    MatToDev(mA);
  } else {
    printf("GPU acceleration disabled!");
  }

  const char* orderStr = "<error>";
  switch (convOrder) {
    case 1: orderStr = "linear";    break;
    case 2: orderStr = "quadratic"; break;
    case 3: orderStr = "cubic";     break;
    case 4: orderStr = "quartic";   break;
  }
  debug("inverting %s with %s convergence...",
        randDim ? "random matrix" : optarg, orderStr);

  if (!softMode) {
    cublasInit();
  }

  Mat mR = 0;
  altmanInvert(mA, &mR, convOrder, errLimit, msLimit, convRateLimit);

  if (MatElemSize(mA) == 2) {
    MatPromote(mA);
  }

  if (outPath) { // optionally write inverted matrix
    MatToHost(mR); // if inverse is on the GPU, download it
    MatWrite(mR, outPath);
    free(outPath);
  }

  // cleanup
  MatFree(mA);
  MatFree(mR);

  if (!softMode) {
    cublasShutDown();
  }

  return 0;
}
