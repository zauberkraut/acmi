// util.c

#include <math.h>
#include <stdarg.h>
#include "invmat.h"
#include "mmio.h"

#define OPTSTR(b, s) (b) ? (s) : ""

static double SPARSE_THRESHOLD = 0.5;

static bool verbose = true;
void setVerbose(bool b) {
  verbose = b;
}

void debug(const char* msg, ...) {
  if (verbose) {
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    fprintf(stderr, "\n");
    va_end(args);
  }
}

void warn(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  fprintf(stderr, "WARNING: ");
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
  va_end(args);
}

void fatal(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

MatInfo readMatInfo(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) fatal("couldn't open %s", path);
  debug("querying %s", path);

  MatInfo info = {0};
  MM_typecode matCode;
  int nCols;
  if (mm_read_banner(f, &matCode)) {
    fatal("couldn't read MatrixMarket banner");
  }
  if (!mm_is_valid(matCode)) {
    fatal("invalid matrix");
  }
  if (mm_is_complex(matCode) || mm_is_hermitian(matCode)) {
    fatal("complex matrices not yet supported");
  }

  info.sparse = mm_is_sparse(matCode);
  info.binary = mm_is_pattern(matCode);
  info.skew = mm_is_skew(matCode);
  info.symmetric = mm_is_symmetric(matCode) | info.skew;

  if (!info.sparse && !mm_is_general(matCode)) {
    fatal("symmetric array encodings not yet supported");
  }

  int err = info.sparse ?
    mm_read_mtx_crd_size(f, (int*)&info.n, &nCols, (int*)&info.nEntries) :
    mm_read_mtx_array_size(f, (int*)&info.n, &nCols);
  if (err) {
    fatal("couldn't read matrix dimensions");
  }
  if (info.n != nCols || info.n < 2) {
    fatal("matrix is %dx%d; only square matrices are invertible", info.n,
          nCols);
  }
  if (info.n > MAX_MAT_DIM) {
    fatal("matrix exceeds maximum-allowed dimension of %d", MAX_MAT_DIM);
  }

  uint64_t n2 = (uint64_t)info.n*info.n;
  info.size = n2*sizeof(float);
  if (info.symmetric) {
    info.nNonzero = 2*info.nEntries - info.n;
  } else info.nNonzero = info.nEntries;

  debug("matrix is %dx%d and %lf MiB in size", info.n, info.n,
        mibibytes(info.size));

  if (info.sparse) {
    info.sparsity = (double)info.nNonzero/n2;
    debug("...and is sparse%s%s with %d nonzero elements and sparsity %g",
          OPTSTR(info.symmetric, ", symmetric"), OPTSTR(info.skew, ", skew"),
          info.nNonzero, info.sparsity);
  }

  fclose(f);
  info.path = path;
  return info;
}

void loadMat(Mat mA, MatInfo info) {
  FILE* f = fopen(info.path, "r");
  debug("loading %s ...", info.path);

  MM_typecode dummyCode;
  int dummy;
  mm_read_banner(f, &dummyCode);
  if (info.sparse) mm_read_mtx_crd_size(f, &dummy, &dummy, &dummy);
  else             mm_read_mtx_array_size(f, &dummy, &dummy);

  Mat hostA = mA->dev ? hostNewMat(info.n) : mA;
  clearMat(hostA);

  if (info.sparse) {
    const char* parseStr = info.binary ? "%d %d\n" : "%d %d %f\n";
    const int paramsPerElem = info.binary ? 2 : 3;
    const int symmSign = info.skew ? -1 : 1;

    for (int i = 0; i < info.nEntries; ++i) {
      int row, col;
      float e = 1;
      if (fscanf(f, parseStr, &row, &col, &e) != paramsPerElem) {
        fatal("error reading element %d from %s", i + 1, info.path);
      }
      setElem(hostA, row - 1, col - 1, e);

      if (info.symmetric && row != col) {
        setElem(hostA, info.n - row, info.n - col, symmSign*e);
      }
    }
  } else {
    for (int col = 0; col < info.n; ++col) {
      for (int row = 0; row < info.n; ++row) {
        float e;
        fscanf(f, "%f", &e);
        setElem(hostA, row, col, e);
      }
    }
  }

  fclose(f);

  if (mA->dev) {
    cuUpload(mA->p, hostA->p, hostA->size);
    freeMat(hostA);
  }
}

void writeMat(FILE* fout, Mat mA) {
  MM_typecode matCode;
  mm_initialize_typecode(&matCode);
  mm_set_matrix(&matCode);
  mm_set_real(&matCode);
  const int n = mA->n;

  Mat hostA;
  if (mA->dev) {
    hostA = hostNewMat(n);
    cuDownload(hostA->p, mA->p, mA->size);
  } else {
    hostA = mA;
  }

  int nNonzero = 0;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (fabsf(elem(hostA, row, col)) > 0) ++nNonzero;
    }
  }
  double sparsity = (double)nNonzero/hostA->n2;
  bool sparse = sparsity < SPARSE_THRESHOLD;

  debug("writing %dx%d matrix with %d nonzero entries; sparsity %g; %s format",
        n, n, nNonzero, sparsity, sparse ? "coordinate" : "array");

  if (sparse) mm_set_coordinate(&matCode);
  else        mm_set_array(&matCode);
  mm_write_banner(fout, matCode);

  if (sparse) {
    mm_write_mtx_crd_size(fout, n, n, nNonzero);

    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        float e = elem(hostA, row, col);
        if (fabsf(e) > 0) {
          fprintf(fout, "%d %d %g\n", row + 1, col + 1, e);
        }
      }
    }
  } else {
    mm_write_mtx_array_size(fout, n, n);
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        fprintf(fout, "%g\n", elem(hostA, row, col));
      }
    }
  }

  if (mA->dev) freeMat(hostA);
}

void saveMat(const char* path, Mat mA) {
  if (path) {
    if (mA->dev) fatal("only host matrices may be saved");
    FILE* f = fopen(path, "w");
    if (!f) fatal("couldn't open %s to write random matrix", path);
    writeMat(f, mA);
    fclose(f);
  }
}

void genRandIntMat(Mat mA, const char* path) {
  const int n = mA->n;
  Mat hostA = mA->dev ? hostNewMat(n) : mA;

  for (int row = 0; row < n; ++row) {
    int64_t rowSum = 0;
    //int signD = rand() % 2 ? 1 : -1;

    for (int col = 0; col < n; ++col) {
      if (row != col) {
        int signE = rand() % 2 ? 1 : -1;
        int r = rand() % n;
        setElem(hostA, row, col, signE*r);
        rowSum += r;
      }
    }

    ++rowSum;
    setElem(hostA, row, row, /*signD*/rowSum);
  }

  saveMat(path, hostA);
  if (mA->dev) {
    cuUpload(mA->p, hostA->p, hostA->size);
    freeMat(hostA);
  }
}

void genRandRealMat(Mat mA, const char* path) {
  const int n = mA->n;
  Mat hostA = mA->dev ? hostNewMat(n) : mA;

  for (int row = 0; row < n; ++row) {
    uint64_t rowSum = 0;
    //int signD = rand() % 2 ? 1 : -1;

    for (int col = 0; col < n; ++col) {
      if (row != col) {
        int signE = rand() % 2 ? 1 : -1;
        int r = rand();
        setElem(hostA, row, col, signE*r/(double)RAND_MAX*n);
        rowSum += r;
      }
    }

    double d = rowSum/(double)RAND_MAX*n + 1;
    setElem(hostA, row, row, /*signD**/d);
  }

  saveMat(path, hostA);
  if (mA->dev) {
    cuUpload(mA->p, hostA->p, hostA->size);
    freeMat(hostA);
  }
}
