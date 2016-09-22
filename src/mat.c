/* mat.c

   ACMI matrix type with functions. */

#include <math.h>
#include "acmi.h"
#include "mmio.h"

/* A matrix is sparse when its proportion of nonzero entries exceeds this. */
static const double SPARSITY_THRESHOLD = 0.05;

struct Mat_ {
  int n;        // square matrix dimension
  int64_t n2;   // n*n
  int elemSize; // bytes per elem
  size_t size;  // total bytes allocated to elems
  size_t pitch; // bytes per row
  void* elems;  // the linear array of matrix entries
  bool dev;     // is matrix in device memory?
  double trace; // the sum of the diagonal entries
};

double ElemVal(union Elem* e, int size) {
  double val = INFINITY;
  switch (size) {
  case 2:  val = halfToSingle(e->fp16); break;
  case 4:  val = e->fp32;               break;
  case 8:  val = e->fp64;               break;
  case 16: val = e->fp64x2.hi;          break; // TODO;
  }
  return val;
}

void ElemSet(union Elem* e, int size, double val) {
  switch (size) {
  case 2:  e->fp16 = singleToHalf(val);          break;
  case 4:  e->fp32 = val;                        break;
  case 8:  e->fp64 = val;                        break;
  case 16: e->fp64x2.hi = val; e->fp64x2.lo = 0; break;
  }
}

/* Initializes a matrix without allocating element space. */
static Mat MatEmptyNew(int n, int elemSize) {
  Mat m = malloc(sizeof(struct Mat_));
  memset(m, 0, sizeof(struct Mat_));
  m->n = n;
  m->n2 = (int64_t)n*n;
  m->elemSize = elemSize;
  m->size = m->n2 * m->elemSize;
  m->pitch = n * m->elemSize;
  m->trace = NAN;
  return m;
}

/* Allocates space for a matrix' elements. */
static void MatNewElems(Mat m, bool dev) {
  m->dev = dev;
  m->elems = dev ? cuMalloc(m->size) : malloc(m->size);
}

/* Makes a new, functional matrix with undefined entry values. */
Mat MatNew(int n, int elemSize, bool dev) {
  Mat m = MatEmptyNew(n, elemSize);
  MatNewElems(m, dev);
  return m;
}

/* Same as MatNew(), but sources parameters from a given template matrix. */
Mat MatBuild(Mat m) {
  return MatNew(m->n, m->elemSize, m->dev);
}

/* Frees a matrix' elements, but not the matrix struct itself. */
static void MatFreeElems(Mat m) {
  assert(m->elems);
  m->dev ? cuFree(m->elems) : free(m->elems);
  m->elems = 0;
}

/* Frees a matrix as well as its elements. */
void MatFree(Mat m) {
  assert(m);

  MatFreeElems(m);
  memset(m, 0, sizeof(struct Mat_));
  free(m);
}

/* Zeroes a matrix' entries. */
void MatClear(Mat m) {
  m->dev ? cuClear(m->elems, m->size) : memset(m->elems, 0, m->size);
}

/* Computes a pointer to any given element of a matrix. */
static inline void* elemAddr(Mat m, int row, int col) {
  unsigned char* p = m->elems;
  return p + col*m->pitch + row*m->elemSize;
}

// getters
int MatN(Mat m) { return m->n; }
int64_t MatN2(Mat m) { return m->n2; }
int MatElemSize(Mat m) { return m->elemSize; }
size_t MatSize(Mat m) { return m->size; }
size_t MatPitch(Mat m) { return m->pitch; }
void* MatElems(Mat m) { return m->elems; }
void* MatCol(Mat m, int col) { return elemAddr(m, 0, col); }
bool MatDev(Mat m) { return m->dev; }
double MatTrace(Mat m) { return m->trace; }

/* Uploads a matrix' elements to device memory, freeing its host memory. */
void MatToDev(Mat m) {
  if (!m->dev) {
    debug("uploading matrix to device");
    void* hostElems = m->elems;
    cuPin(hostElems, m->size);
    MatNewElems(m, true);
    cuUpload(m->elems, hostElems, m->size);
    cuUnpin(hostElems);
    free(hostElems);
  }
}

/* Downloads a matrix' elements to host memory, freeing its device memory. */
void MatToHost(Mat m) {
  if (m->dev) {
    debug("downloading matrix from device");
    void* devElems = m->elems;
    MatNewElems(m, false);
    cuPin(m->elems, m->size);
    cuDownload(m->elems, devElems, m->size);
    cuUnpin(m->elems);
    cuFree(devElems);
  }
}

/* Converts a 32-bit matrix to 64-bit. */
void MatPromote(Mat m) {
  assert(m->elemSize < MAX_ELEM_SIZE);

  struct Mat_ mOrig = *m;
  m->elemSize <<= 1; // TODO: double double
  m->size     <<= 1;
  m->pitch    <<= 1;
  MatNewElems(m, m->dev);

  if (m->dev) {
    cuPromote(m->elems, mOrig.elems, mOrig.elemSize, mOrig.n2);
  } else {
    for (int64_t i = 0; i < mOrig.n2; i++) {
      switch (mOrig.elemSize) {
      case 4: ((double*)m->elems)[i] = ((float*)mOrig.elems)[i]; break;
      case 8: fatal("WIP"); break;
      }
    }
  }

  MatFreeElems(&mOrig);
}

/* Returns a matrix element. */
double MatGet(Mat m, int row, int col) {
  assert(!m->dev && row >= 0 && row < m->n && col >= 0 && col < m->n);
  union Elem e;
  m->dev ? cuDownload(&e, elemAddr(m, row, col), m->elemSize)
         : memcpy(&e, elemAddr(m, row, col), m->elemSize);
  return ElemVal(&e, m->elemSize);
}

/* Sets a matrix element. */
void MatPut(Mat m, int row, int col, double elem) {
  assert(!m->dev && row >= 0 && row < m->n && col >= 0 && col < m->n);
  union Elem e;
  ElemSet(&e, m->elemSize, elem);
  m->dev ? cuUpload(elemAddr(m, row, col), &e, m->elemSize)
         : memcpy(elemAddr(m, row, col), &e, m->elemSize);
}

/* Loads a matrix of the given precision from a file. */
Mat MatLoad(const char* path, int elemSize, bool attrOnly) {
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

  int64_t n2 = (int64_t)n*n;
  size_t size = n2*elemSize;
  debug("matrix is %dx%d and %.3f MiB in size", n, n, mibibytes(size));

  double sparsity = INFINITY;
  if (coord) {
    if (symmOrSkew) {
      debug("...and is %ssymmetric", skew ? "skew-" : "");
    }

    int64_t nNonzero = symmOrSkew ? 2*nEntries - n : nEntries;
    sparsity = (double)nNonzero/n2;
    debug("...and has %ld nonzero elements and sparsity %g", nNonzero,
          sparsity);
    if (sparsity < SPARSITY_THRESHOLD) {
      debug("...and qualifies as sparse");
    }
  }

  if (attrOnly) { // user just wanted the file specs reported above; return
    fclose(in);
    return 0;
  }

  Mat m = MatNew(n, elemSize, false);
  m->trace = 0; // we'll compute the trace, so clear the NaN
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
        MatPut(m, col - 1, row - 1, symmSign*elem);
      } else if (row == col) {
        m->trace += elem;
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
          MatPut(m, col, row, symmSign*elem);
        } else if (row == col) {
          m->trace += elem;
        }
      }
    }
  }

  fclose(in);

  return m;
}

/* Writes a matrix out to a given path. */
void MatWrite(Mat m, const char* path) {
  if (m->dev) {
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
  const int n = m->n;

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

/* Generates a random, probably-invertible matrix of integers.
   Allowing negative entries shall probably cause ill-conditioning. */
Mat MatNewRand(int n, int elemSize, double maxElem, bool symm, bool real,
               bool neg, bool diagDom) {
  Mat m = MatNew(n, elemSize, false);
  m->trace = 0;
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
        double absElem = real ? (double)rand() / RAND_MAX * maxElem :
                                (double)(rand() % maxElemEx);
        double sign = neg ? (rand() % 2 ? 1 : -1) : 1;
        double elem = sign * absElem;
        MatPut(m, row, col, elem);
        if (symm && row != col) { // mirror symmetric element
          MatPut(m, col, row, elem);
        }
        rowSum += absElem;

        if (row == col) {
          m->trace += elem;
        }
      }
    }

    if (diagDom) {
      double sign = neg ? (rand() % 2 ? 1 : -1) : 1;
      // make diagonal strictly greater than the sum of the other row entries
      double diag = sign * (real ? nextafter(rowSum, INFINITY) : rowSum + 1);
      MatPut(m, row, row, diag);
      m->trace += diag;
    }
  }

  return m;
}

void MatDebug(Mat m) {
  debug("matrix is: %dx%dx%d = %.3f MiB, %ld elements, %ld bytes per col\n"
        "dev: %d, trace %g", m->n, m->n, m->elemSize, mibibytes(m->size), m->n2,
        m->pitch, m->dev, m->trace);
  for (int row = 0; row < m->n; row++) {
    for (int col = 0; col < m->n; col++) {
      printf("%.3f ", MatGet(m, row, col));
    }
    printf("\n");
  }
}
