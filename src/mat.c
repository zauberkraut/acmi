// mat.c

#include "invmat.h"

static uint64_t g_devTotalMatBytes = 0;
static uint64_t g_hostTotalMatBytes = 0;

static Mat emptyMat(int n) {
  Mat m = malloc(sizeof(Mat_));
  memset(m, 0, sizeof(Mat_));
  m->n = n;
  m->n2 = (uint64_t)n * n;
  m->size = m->n2 * sizeof(float);
  return m;
}

Mat hostNewMat(int n) {
  Mat m = emptyMat(n);
  m->dev = false;
  m->p = malloc(m->size);

  g_hostTotalMatBytes += m->size;
  debug("allocated %.3lf MiB to %dx%d matrix on host; %.3lf MiB total",
        mibibytes(m->size), n, n, mibibytes(g_hostTotalMatBytes));
  return m;
}

Mat devNewMat(int n) {
  Mat m = emptyMat(n);
  m->dev = true;
  m->p = cuMalloc(m->size);

  g_devTotalMatBytes += m->size;
  debug("allocated %.3lf MiB to %dx%d matrix on device; %.3lf MiB total",
        mibibytes(m->size), n, n, mibibytes(g_devTotalMatBytes));
  return m;
}

void clearMat(Mat m) {
  if (m->dev) cuClear(m->p, m->size);
  else           memset(m->p, 0, m->size);
}

void freeMat(Mat m) {
  if (!m || !m->p) {
    warn("attempted to free null or empty matrix");
  } else {
    if (m->dev) {
      cuFree(m->p);
      g_devTotalMatBytes -= m->size;
    } else {
      free(m->p);
      g_hostTotalMatBytes -= m->size;
    }
    m->p = 0;
    debug("freed %.3lf MiB from %dx%d matrix on %s; %.3lf MiB remain",
          mibibytes(m->size), m->n, m->n, m->dev ? "device" : "host",
          mibibytes(m->dev ? g_devTotalMatBytes : g_hostTotalMatBytes));
    free(m);
  }
}

static void bound(Mat m, int row, int col) {
  if (row < 0 || row > m->n || col < 0 || col > m->n) {
    fatal("attempted to access element (%d, %d) of a %dx%d matrix", row, col,
          m->n, m->n);
  }
}

float elem(Mat m, int row, int col) {
  bound(m, row, col);
  float e;
  if (m->dev) {
    cuDownload(&e, m->p + col*m->n + row, sizeof(e));
  } else e = m->p[col*m->n + row];
  return e;
}

void setElem(Mat m, int row, int col, float e) {
  bound(m, row, col);
  if (m->dev) {
    cuUpload(m->p + col*m->n + row, &e, sizeof(e));
  } else m->p[col*m->n + row] = e;
}
