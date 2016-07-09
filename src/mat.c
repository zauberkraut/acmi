// mat.c

#include <math.h>
#include "acmi.h"
#include "mmio.h"

static const double SPARSITY_THRESHOLD = 0.05;

struct Mat_ {
  int n;
  int64_t n2;
  size_t size;
  float* elements;
  bool dev;
  bool symmetric;
  bool sparse;
  double trace;
};

static size_t g_devTotalMatBytes = 0;
static size_t g_hostTotalMatBytes = 0;

static Mat MatEmptyNew(int n) {
  Mat m = malloc(sizeof(struct Mat_));
  memset(m, 0, sizeof(struct Mat_));
  m->n = n;
  m->n2 = (int64_t)n*n;
  m->size = m->n2*sizeof(float);
  m->trace = NAN;
  return m;
}

static void updateTotalMatBytes(Mat m, bool alloc) {
  size_t* total = m->dev ? &g_devTotalMatBytes : &g_hostTotalMatBytes;
  if (alloc) *total += m->size; else *total -= m->size;
  debug("%s %.3f MiB %dx%d matrix on %s; %.3f MiB %s",
        alloc ? "allocating" : "freeing", mibibytes(m->size), m->n, m->n,
        m->dev ? "device" : "host", mibibytes(*total),
        alloc ? "total" : "remain");
}

static void MatNewElements(Mat m, bool dev) {
  m->dev = dev;
  m->elements = dev ? cuMalloc(m->size) : malloc(m->size);
  updateTotalMatBytes(m, true);
}

Mat MatNew(int n, bool dev) {
  Mat m = MatEmptyNew(n);
  MatNewElements(m, dev);
  return m;
}

static void MatFreeElements(Mat m) {
  assert(m->elements);

  if (m->dev) cuFree(m->elements);
  else        free(m->elements);
  m->elements = 0;
  updateTotalMatBytes(m, false);
}

void MatFree(Mat m) {
  assert(m);

  MatFreeElements(m);
  memset(m, 0, sizeof(struct Mat_));
  free(m);
}

void MatClear(Mat m) {
  if (m->dev) cuClear(m->elements, m->size);
  else        memset(m->elements, 0, m->size);
}

int MatN(Mat m) { return m->n; }
int64_t MatN2(Mat m) { return m->n2; }
size_t MatSize(Mat m) { return m->size; }
float* MatElements(Mat m) { return m->elements; }
bool MatDev(Mat m) { return m->dev; }

double MatTrace(Mat m) {
  if (m->trace == NAN) {
    m->trace = 0;
    for (int i = 0; i < m->n; ++i) {
      m->trace += MatGet(m, i, i);
    }
  }

  return m->trace;
}

void MatToDev(Mat m) {
  if (!m->dev) {
    debug("uploading matrix to device");
    updateTotalMatBytes(m, false);
    float* hostElements = m->elements;
    MatNewElements(m, true);
    cuUpload(m->elements, hostElements, m->size);
    free(hostElements);
  }
}

void MatToHost(Mat m) {
  if (m->dev) {
    debug("downloading matrix from device");
    updateTotalMatBytes(m, false);
    float* devElements = m->elements;
    MatNewElements(m, false);
    cuDownload(m->elements, devElements, m->size);
    cuFree(devElements);
  }
}

float MatGet(Mat m, int row, int col) {
  assert(row >= 0 && row < m->n && col >= 0 && col < m->n);
  float e;
  if (m->dev) cuDownload(&e, m->elements + col*m->n + row, sizeof(e));
  else        e = m->elements[col*m->n + row];
  return e;
}

void MatPut(Mat m, int row, int col, float e) {
  assert(row >= 0 && row < m->n && col >= 0 && col < m->n);
  if (m->dev) cuUpload(m->elements + col*m->n + row, &e, sizeof(e));
  else        m->elements[col*m->n + row] = e;
}

Mat MatLoad(const char* path, bool attrOnly) {
  FILE* in = fopen(path, "r");
  if (!in) fatal("couldn't open %s", path);

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

  if (!coord && !mm_is_general(matCode)) {
    fatal("symmetric array encodings not yet supported");
  }

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
  size_t size = n2*sizeof(float);
  debug("matrix is %dx%d and %.3f MiB in size", n, n, mibibytes(size));

  double sparsity = INFINITY;
  if (coord) {
    if (symmOrSkew) debug("...and is %ssymmetric", skew ? "skew-" : "");

    int64_t nNonzero = symmOrSkew ? 2*nEntries - n : nEntries;
    sparsity = (double)nNonzero/n2;
    debug("...and has %ld nonzero elements and sparsity %g", nNonzero,
          sparsity);
    if (sparsity < SPARSITY_THRESHOLD) debug("...and qualifies as sparse");
  }

  if (attrOnly) {
    fclose(in);
    return 0;
  }

  Mat m = MatNew(n, false);
  m->symmetric = mm_is_symmetric(matCode);
  m->sparse = sparsity < SPARSITY_THRESHOLD;

  if (coord) {
    MatClear(m);
    const char* parseStr    = mm_is_pattern(matCode) ? "%d %d\n" : "%d %d %f\n";
    const int paramsPerElem = mm_is_pattern(matCode) ? 2 : 3;
    const int symmSign = skew ? -1 : 1;

    for (int i = 0; i < nEntries; ++i) {
      int row, col;
      float e = 1;
      if (fscanf(in, parseStr, &row, &col, &e) != paramsPerElem) {
        fatal("error reading element %d from %s", i + 1, path);
      }
      MatPut(m, row - 1, col - 1, e);

      if (symmOrSkew && row != col) {
        MatPut(m, n - row, n - col, symmSign*e);
      }
    }
  } else {
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        float e;
        fscanf(in, "%f", &e);
        MatPut(m, row, col, e);
      }
    }
  }

  fclose(in);

  MatTrace(m); // compute trace

  return m;
}

void MatWrite(Mat m, const char* path) {
  if (m->dev) fatal("only host matrices may be written to disk");
  FILE* out = fopen(path, "w");
  if (!out) fatal("couldn't open %s to write random matrix", path);

  MM_typecode matCode;
  mm_initialize_typecode(&matCode);
  mm_set_matrix(&matCode);
  mm_set_real(&matCode);
  const int n = m->n;

  int64_t nNonzero = 0;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (fabsf(MatGet(m, row, col)) > 0) ++nNonzero;
    }
  }
  double sparsity = (double)nNonzero/m->n2;
  bool sparse = sparsity < SPARSITY_THRESHOLD;

  debug("writing %dx%d matrix with %ld nonzero entries; sparsity %g; %s format",
        n, n, nNonzero, sparsity, sparse ? "coordinate" : "array");

  if (sparse) mm_set_coordinate(&matCode);
  else        mm_set_array(&matCode);

  mm_write_banner(out, matCode);

  if (sparse) {
    mm_write_mtx_crd_size(out, n, n, nNonzero);

    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        float e = MatGet(m, row, col);
        if (fabsf(e) > 0) {
          fprintf(out, "%d %d %g\n", row + 1, col + 1, e);
        }
      }
    }
  } else {
    mm_write_mtx_array_size(out, n, n);

    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        fprintf(out, "%g\n", MatGet(m, row, col));
      }
    }
  }

  fclose(out);
}

Mat MatRandDiagDom(int n, bool real, bool symmetric) {
  if (symmetric) assert(false);
  Mat m = MatNew(n, false);

  for (int row = 0; row < n; ++row) {
    int64_t rowSum = 0;
    //int signD = rand() % 2 ? 1 : -1;

    for (int col = 0; col < n; ++col) {
      if (row != col) {
        int signE = rand() % 2 ? 1 : -1;
        int r = rand();
        if (!real) r %= n;
        float v = signE*(real ? r*n/(double)RAND_MAX : r);
        MatPut(m, row, col, v);
        rowSum += r;
      }
    }

    float v = 1 + (real ? rowSum*n/(double)RAND_MAX : rowSum);
    MatPut(m, row, row, /*signD*/v);
  }

  return m;
}
