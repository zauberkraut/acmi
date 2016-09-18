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

const int MAX_PATH_LEN = 255;
const int MIN_ELEM_BITS = 32;
const int MAX_ELEM_BITS = 64;
const int DEFAULT_ELEM_SIZE = 4;
const double DEFAULT_ERR_LIMIT = 0.00001;
const int DEFAULT_MS_LIMIT = 60000; // one minute
const int MAX_MS_LIMIT = 86400000;  // one day

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
         "Options:\n"
         "  -h          These instructions\n"
         "  -q          Disable logging\n"
         "  -i          Print matrix file info and exit\n"
         "  -o <path>   Output computed matrix inverse to path\n"
         "  -p <#bits>  Set floating-point matrix element precision (32 or 64, default: %d)\n"
         "  -2          Employ quadratic instead of cubic convergence\n"
         "  -e <real>   Set inversion error limit (default: %g)\n"
         "  -t <ms>     Set inversion time limit in ms (default: %d ms)\n"
         "  -S          Perform all computations in software without the GPU.\n"
         "Random matrix options:\n"
         "  -s          Generate symmetric matrix\n"
         "  -O <path>   Output generated, uninverted random matrix to path\n"
         "  -x <hex>    Set PRNG seed\n"
         "Currently, only Matrix Market files are are supported.\n"
         "To generate and invert a diagonally-dominant random matrix, enter the matrix\n"
         "dimension prefixed by '?' instead of a filename.\n\n",
         8*DEFAULT_ELEM_SIZE, DEFAULT_ERR_LIMIT, DEFAULT_MS_LIMIT);
  exit(0);
}

int main(int argc, char* argv[]) {
  bool infoMode = false;
  char* outPath = 0;
  bool softMode = false;
  int elemSize = DEFAULT_ELEM_SIZE;
  bool quadConv = false;
  double errLimit = DEFAULT_ERR_LIMIT;
  int msLimit = DEFAULT_MS_LIMIT;
  int randDim = 0;
  bool randSymm = false;
  char* randOutPath = 0;
  unsigned prngSeed = 0;

  if (1 == argc) {
    usage();
  }

  opterr = 0;
  int opt;
  while ((opt = getopt(argc, argv, "hqio:p:2e:t:SsO:x:")) != -1) {
    switch (opt) {
      int elemBits;
      double pow2;

    case 'h': usage();           break;
    case 'q': setVerbose(false); break;
    case 'i': infoMode = true;   break;
    case '2': quadConv = true;   break;
    case 'S': softMode = true;   break;
    case 's': randSymm = true;   break;

    case 'o':
      checkWriteAccess(optarg);
      outPath = strndup(optarg, MAX_PATH_LEN);
      break;
    case 'p':
      elemBits =
        (int)parseInt(10, MIN_ELEM_BITS, MAX_ELEM_BITS,
                      "invalid floating-point precision");
      pow2 = log2(elemBits);
      if (pow2 != floor(pow2)) {
        fatal("invalid floating-point precision");
      }
      elemSize = elemBits/8;
      break;
    case 'e':
      errLimit = parseFloat(0, 1,
                            "error limit must be a real on [0, 1)");
      break;
    case 't':
      msLimit = (int)parseInt(10, 0, MAX_MS_LIMIT, "invalid time limit");
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

  Mat mA = 0;

  if (!softMode) {
    debug("%g MiB device memory available", mibibytes(cuMemAvail()));
  }

  if (randDim) { // random mode
    if (!prngSeed) prngSeed = time(0); // if user supplied no seed, use time
    debug("seeding PRNG with %x", prngSeed);
    srand(prngSeed);

    debug("generating %d-bit random %dx%d%s matrix...", 8*elemSize,
          randDim, randDim, randSymm ? " symmetric" : "");
    mA = MatRandDiagDom(randDim, elemSize, randSymm);

    if (randOutPath) { // optionally write randomly-generated matrix
      MatWrite(mA, randOutPath);
      free(randOutPath);
    }
  } else { // load matrix from given file
    debug("loading %s", optarg);
    mA = MatLoad(optarg, elemSize, infoMode);
  }

  const double matMiB = mibibytes(MatSize(mA));
  debug("%g MiB/matrix; allocating %g MiB total", matMiB, 4*matMiB);

  if (infoMode) {
    debug("matrix info-only mode: terminating");
    exit(0);
  }

  if (!softMode) { // upload source matrix to GPU
    MatToDev(mA);
  }

  const char* zusatz = "";
  if (quadConv) {
    zusatz = "quadratically ";
  }
  debug("%sinverting %s...", zusatz, randDim ? "random matrix" : optarg);

  if (!softMode) {
    initCublas();
  }

  Mat mR = 0;
  altmanInvert(mA, &mR, errLimit, msLimit, quadConv);

  if (outPath) { // optionally write inverted matrix
    MatToHost(mR); // if inverse is on the GPU, download it
    MatWrite(mR, outPath);
    free(outPath);
  }

  // cleanup
  MatFree(mA);
  MatFree(mR);

  if (!softMode) {
    shutDownCublas();
  }

  return 0;
}
