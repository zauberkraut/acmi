/* util.c

   ACMI utility functions. */

#include <assert.h>
#include <cpuid.h>
#include <immintrin.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include "acmi.h"
#include "mmio.h"

enum { MAT_PRINT_EXTENT = 8 };
/* A matrix is sparse when its proportion of nonzero entries exceeds this. */
static const double SPARSITY_THRESHOLD = 0.05;

static void print(FILE* f, const char* s, va_list args) {
  vfprintf(f, s, args);
  fprintf(f, "\n");
  fflush(f);
  va_end(args);
}

void debug(const char* s, ...) {
  va_list args; va_start(args, s); print(stdout, s, args);
}

void warn(const char* s, ...) {
  fprintf(stderr, "WARNING: ");
  va_list args; va_start(args, s);
  print(stderr, s, args);
}

void fatal(const char* s, ...) {
  fprintf(stderr, "FATAL: ");
  va_list args; va_start(args, s);
  print(stderr, s, args);
  abort();
}

double mibibytes(size_t size) {
  return (double)size/(1 << 20);
}

void checkDevMemEnough(int n, int elemSize, int matCount) {
  const size_t totalSize = elemSize * n * n * matCount;
  const size_t available = cuMemAvail();
  if (totalSize > available) {
    fatal("%ld bytes device memory needed; only %ld available", totalSize,
          available);
  }
}

/* Tests for CPU support of the RDRAND instruction. */
static bool rdRandSupported() {
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ecx & bit_RDRND;
}

static int cstdRand16() {
  return (int)((unsigned)rand() >> 15);
}

/* Uses RDRAND instruction to generate high-quality random integers.
   Intended for use in the creation of k-sorted sequences for a given k.
   Requires an Ivy Bridge or newer x86 CPU. Requires no seeding. */
static int rdRand16() {
  int r = 0;
  if (!_rdrand16_step((uint16_t*)&r)) {
    fatal("RDRAND ran out of entropy; sourcing from rand()");
  }
  return r;
}

/* Loads a matrix of the given precision from a file. */
Mat MatLoad(const char* path, int elemSize, int matCount) {
  FILE* in = fopen(path, "r");
  if (!in) {
    fatal("couldn't open %s", path);
  }

  MM_typecode matCode;
  if (mm_read_banner(in, &matCode)) {
    fatal("couldn't read Matrix Market banner");
  }
  if (!mm_is_valid(matCode)) {
    fatal("invalid matrix");
  }
  if (mm_is_complex(matCode) || mm_is_hermitian(matCode)) {
    fatal("complex matrices not yet supported");
  }

  bool coord = mm_is_coordinate(matCode);
  bool skew = mm_is_skew(matCode);
  bool symmOrSkew = mm_is_symmetric(matCode) | skew;

  int n, nCols, nEntries;
  int err = coord ? mm_read_mtx_crd_size(in, &n, &nCols, &nEntries) :
                    mm_read_mtx_array_size(in, &n, &nCols);
  if (err) {
    fatal("couldn't read matrix dimensions");
  }
  if (n != nCols || n < 2) {
    fatal("matrix is %dx%d; only square matrices are invertible", n, nCols);
  }
  if (n > MAX_MAT_DIM) {
    fatal("matrix exceeds maximum-allowed dimension of %d", MAX_MAT_DIM);
  }
  if (matCount) {
    checkDevMemEnough(n, elemSize, matCount);
  }

  int64_t n2 = (int64_t)n*n;
  size_t size = n2*elemSize;
  debug("matrix is %dx%d and %.3f MiB in size", n, n, mibibytes(size));

  double sparsity = INFINITY;
  if (coord) {
    if (symmOrSkew) {
      debug("...and is %ssymmetric", skew ? "skew-" : "");
    }

    int64_t nNonzero = symmOrSkew ? 2 * nEntries - n : nEntries;
    sparsity = (double)nNonzero/n2;
    debug("...and has %ld nonzero elements and sparsity %g", nNonzero,
          sparsity);
    if (sparsity < SPARSITY_THRESHOLD) {
      debug("...and qualifies as sparse");
    }
  }

  Mat m = MatNew(n, elemSize, false);
  const int symmSign = skew ? -1 : 1;

  if (coord) {
    MatClear(m);
    const char* parseStr = mm_is_pattern(matCode) ? "%d %d\n" : "%d %d %lf\n";
    const int paramsPerElem = mm_is_pattern(matCode) ? 2 : 3;

    for (int i = 0; i < nEntries; i++) {
      int row, col;
      double elem = 1;
      if (fscanf(in, parseStr, &row, &col, &elem) != paramsPerElem) {
        fatal("error reading element %d from %s", i + 1, path);
      }
      MatPut(m, row - 1, col - 1, elem);

      if (symmOrSkew && row != col) { // mirror symmetric element
        MatPut(m, col - 1, row - 1, symmSign * elem);
      }
    }
  } else { // dense array encoding
    for (int col = 0; col < n; col++) {
      int row = symmOrSkew ? col : 0;

      for (; row < n; row++) {
        double elem;
        fscanf(in, "%lf\n", &elem);
        MatPut(m, row, col, elem);

        if (symmOrSkew && row != col) {
          MatPut(m, col, row, symmSign * elem);
        }
      }
    }
  }

  fclose(in);

  MatTrace(m); // compute trace
  return m;
}

static int drawSign(int* n) {
  int tmp = *n;
  *n >>= 1;
  return (tmp & 1) ? 1 : -1;
}

/* Generates a random, probably-invertible matrix of integers.
   Allowing negative entries shall probably cause ill-conditioning.
   TODO: replace bool cascade with ORed flags
   TODO: diagonal dominance adjustment of +1 might be swallowed for large values
   TODO: 16-bit reals are choppy */
Mat MatNewRand(int n, int elemSize, double maxElem, bool symm, bool real,
               bool neg, bool diagDom, bool useHardwareRNG) {
  if (useHardwareRNG && !rdRandSupported()) {
    fatal("your CPU doesn't support the RDRAND instruction");
  }
  int (*rand16)() = useHardwareRNG ? rdRand16 : cstdRand16;
  int rMax = neg ? SHRT_MAX : USHRT_MAX;

  Mat m = MatNew(n, elemSize, false);
  int maxElemEx = floor(maxElem) + 1;

  for (int row = 0; row < n; row++) {
    int col = symm ? row : 0;
    double rowSum = 0;
    // if symmetric, sum the elements before this diagonal
    for (int i = 0; i < col; i++) {
      rowSum += MatGet(m, row, i);
    }

    for (; col < n; col++) {
      if (!diagDom || row != col) { // diagonals are set below from the computed sum
        int r = rand16();
        double sign = neg ? drawSign(&r) : 1;
        double absElem = real ? (double)r / rMax * maxElem :
                                (double)(r % maxElemEx);
        double elem = sign * absElem;
        MatPut(m, row, col, elem);
        if (symm && row != col) { // mirror symmetric element
          MatPut(m, col, row, elem);
        }
        rowSum += absElem;
      }
    }

    if (diagDom) {
      double sign = neg ? (rand16() & 1 ? 1 : -1) : 1;
      // make diagonal strictly greater than the sum of the other row entries
      double diag = sign * (real ? nextafter(rowSum, INFINITY) : rowSum + 1);
      MatPut(m, row, row, diag);
    }
  }

  MatTrace(m); // compute trace
  return m;
}

/* Writes a matrix out to a given path. */
void MatWrite(Mat m, const char* path) {
  if (MatDev(m)) {
    fatal("only host matrices may be written to disk");
  }
  FILE* out = fopen(path, "w");
  if (!out) {
    fatal("couldn't open %s to write random matrix", path);
  }

  MM_typecode matCode;
  mm_initialize_typecode(&matCode);
  mm_set_matrix(&matCode);
  mm_set_real(&matCode);
  mm_set_array(&matCode);
  const int n = MatN(m);

  debug("writing %.3f MiB %dx%d matrix in array format", mibibytes(MatSize(m)),
        n, n);

  mm_write_banner(out, matCode);
  mm_write_mtx_array_size(out, n, n);
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < n; row++) {
      fprintf(out, "%g\n", MatGet(m, row, col));
    }
  }

  fclose(out);
}

void MatPrint(Mat m) {
  assert(!MatDev(m));

  printf("%ld %d-bit elements; %ld bytes per column; trace = %g\n\n",
         MatN2(m), 8 * MatElemSize(m), MatPitch(m), MatTrace(m));
  const int extent = iMin(MAT_PRINT_EXTENT, MatN(m));

  for (int row = 0; row < extent; row++) {
    for (int col = 0; col < extent; col++) {
      printf("%12g ", MatGet(m, row, col));
    }
    if (extent < MatN(m)) {
      printf("...");
    }
    printf("\n");
  }
}
