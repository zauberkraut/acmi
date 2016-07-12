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
const Impl DEFAULT_IMPL = CUBLAS_IMPL;
const double DEFAULT_MAX_ERROR = 0.00001;
const int DEFAULT_MAX_STEP = 16;

extern char *optarg;
extern int optind, opterr, optopt;

/* Parses an integer argument of the given radix from the command line, aborting
   after printing errMsg if an error occurs or the integer exceeds the given
   bounds. */
long parseInt(int radix, long min, long max, const char* errMsg) {
  char* parsePtr = 0;
  long l = strtol(optarg, &parsePtr, radix);
  if (parsePtr - optarg != strlen(optarg) || l < min || l > max ||
      errno == ERANGE) {
    fatal(errMsg);
  }
  return l;
}

double parseFloat(double min, double maxEx, const char* errMsg) {
  char* parsePtr;
  double v = strtof(optarg, &parsePtr);
  if (parsePtr - optarg != strlen(optarg) || v < min || v >= maxEx ||
      errno == ERANGE) {
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
  if (path[len-1] == '/' || path[len-1] == '\\' || (dir = opendir(path))) {
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
         "  -m <mode>   Select implementation to run (default: cublas)\n"
         "  -d          Enable double-precision floating-point matrix elements\n"
         "  -2          Employ quadratic instead of cubic convergence\n"
         "  -e <real>   Set max inversion error (default: %g)\n"
         "  -n <count>  Set max iteration to compute (default: %d)\n"
         "Random matrix options:\n"
         "  -s          Generate symmetric matrix\n"
         "  -O <path>   Output generated, uninverted random matrix to path\n"
         "  -x <hex>    Set PRNG seed\n"
         "Currently, only Matrix Market files are are supported.\n"
         "To generate and invert a diagonally-dominant random matrix, enter the matrix\n"
         "dimension prefixed by '?' instead of a filename.\n\n"
         "Modes available through the -m option:\n"
         "  cpu     Pure software BLAS implementation; no Nvidia GPU needed\n"
         "  cublas  Nvidia CUDA BLAS implementation\n"
         "  lu      Nvidia LU-factorization inversion without iteration\n",
         DEFAULT_MAX_ERROR, DEFAULT_MAX_STEP);
  exit(0);
}

int main(int argc, char* argv[]) {
  bool infoMode = false;
  char* outPath = 0;
  Impl impl = DEFAULT_IMPL;
  bool doublePrec = false;
  bool quadConv = false;
  double maxError = DEFAULT_MAX_ERROR;
  int maxStep = DEFAULT_MAX_STEP;
  int randDim = 0;
  bool randSymm = false;
  char* randOutPath = 0;
  unsigned prngSeed = 0;

  if (argc == 1) {
    usage();
  }

  opterr = 0;
  int opt;
  while ((opt = getopt(argc, argv, "hqio:m:d2e:n:sO:x:")) != -1) {
    switch (opt) {
    case 'h': usage();              break;
    case 'q': setVerbose(false);    break;
    case 'i': infoMode = true;      break;
    case 'd': doublePrec = true;      break;
    case '2': quadConv = true;      break;
    case 's': randSymm = true; break;

    case 'o':
      checkWriteAccess(optarg);
      outPath = strndup(optarg, MAX_PATH_LEN);
      break;
    case 'm':
      if (!strncmp(optarg, "cpu", 8)) {
        impl = CPU_IMPL;
      } else if (!strncmp(optarg, "cublas", 8)) {
        impl = CUBLAS_IMPL;
      } else if (!strncmp(optarg, "lu", 8)) {
        impl = LU_IMPL;
      } else {
        fatal("supported modes are: \"cpu\", \"cublas\", \"lu\"");
      }
      break;
    case 'e':
      maxError = parseFloat(0, 1,
                            "max error measure must be a real on [0, 1)");
      break;
    case 'n':
      maxStep = (int)parseInt(10, 0, 1000, "invalid step limit");
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
  if (!optarg || strlen(optarg) == 0) {
    fatal("missing input file");
  }
  if (optind < argc - 1) {
    fatal("unexpected argument: %s", argv[optind+1]);
  }
  if (optarg[0] == '?') { // random mode
    ++optarg; // parse remainder of argument as the matrix dimension
    randDim = (int)parseInt(10, 2, MAX_MAT_DIM,
                            "invalid random matrix dimension");
  } else {
    if (randSymm || randOutPath || prngSeed) {
      fatal("options -r, -s, -O and -x apply only to random matrices");
    }
  }

  Mat mA = 0;

  if (randDim) { // random mode
    if (!prngSeed) prngSeed = time(0); // if user supplied no seed, use time
    debug("seeding PRNG with %x", prngSeed);
    srand(prngSeed);

    debug("generating %d-bit random %dx%d%s matrix...", doublePrec ? 64 : 32,
          randDim, randDim, randSymm ? " symmetric" : "");
    mA = MatRandDiagDom(randDim, doublePrec, randSymm);

    if (randOutPath) { // optionally write randomly-generated matrix
      MatWrite(mA, randOutPath);
      free(randOutPath);
    }
  } else { // load matrix from given file
    debug("loading %s", optarg);
    mA = MatLoad(optarg, doublePrec, infoMode);
  }

  if (infoMode) {
    debug("matrix info-only mode: terminating");
    exit(0);
  }

  if (impl != CPU_IMPL) { // upload source matrix to GPU
    MatToDev(mA);
  }
  Mat mR = MatBuild(mA); // initialize inverse matrix with the same parameters

  const char* zusatz = "";
  if (quadConv)             zusatz = "quadratically";
  else if (impl == LU_IMPL) zusatz = "using LU factorization";
  debug("inverting %s %s...", randDim ? "random matrix" : optarg, zusatz);

  switch (impl) {
  case CUBLAS_IMPL:
    initCublas();
  case CPU_IMPL:
    // invert using convergent sequence
    altmanInvert(mA, mR, maxError, maxStep, quadConv);
    break;
  case LU_IMPL:
    initCublas();
    debug("inverting using cuBLAS LU-factorization");
    luInvert(mA, mR);
    break;
  }

  if (outPath) { // optionally write inverted matrix
    MatToHost(mR); // if inverse is on the GPU, download it
    MatWrite(mR, outPath);
    free(outPath);
  }

  // cleanup
  MatFree(mA);
  MatFree(mR);

  switch (impl) {
  case CUBLAS_IMPL:
  case LU_IMPL:
    shutDownCublas();
    break;
  default:
    break;
  }

  return 0;
}
