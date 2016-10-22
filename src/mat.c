/* mat.c

   ACMI matrix type with functions. */

#include "acmi.h"

struct Mat_ {
  int n;        // square matrix dimension
  int64_t n2;   // n*n
  int elemSize; // bytes per elem
  size_t size;  // total bytes allocated to elems
  size_t pitch; // bytes per row
  void* elems;  // the linear array of matrix entries
  bool dev;     // is matrix in device memory?
};

static inline double ElemVal(union Elem* e, int size) {
  double val = INFINITY;
  switch (size) {
  case 4:  val = e->fp32; break;
  case 8:  val = e->fp64; break;
  }
  return val;
}

static inline void ElemSet(union Elem* e, int size, double val) {
  switch (size) {
  case 4:  e->fp32 = val; break;
  case 8:  e->fp64 = val; break;
  }
}

/* Initializes a matrix without allocating element space. */
static Mat MatEmptyNew(int n, int elemSize) {
  Mat m = malloc(sizeof(struct Mat_));
  memset(m, 0, sizeof(struct Mat_));
  m->n = n;
  m->n2 = (int64_t)n*n;
  m->elemSize = elemSize;
  m->size = m->n2*m->elemSize;
  m->pitch = n*m->elemSize;
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
  m->dev ? cuFree(m->elems) : free(m->elems);
  m->elems = 0;
}

/* Frees a matrix as well as its elements. */
void MatFree(Mat m) {
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
inline int MatN(Mat m) { return m->n; }
inline int64_t MatN2(Mat m) { return m->n2; }
inline int MatElemSize(Mat m) { return m->elemSize; }
inline size_t MatSize(Mat m) { return m->size; }
inline size_t MatPitch(Mat m) { return m->pitch; }
inline void* MatElems(Mat m) { return m->elems; }
inline bool MatDev(Mat m) { return m->dev; }

/* Uploads a matrix' elements to device memory, freeing its host memory. */
void MatToDev(Mat m) {
  debug("uploading matrix to device");
  void* hostElems = m->elems;
  cuPin(hostElems, m->size);
  MatNewElems(m, true);
  cuUpload(m->elems, hostElems, m->size);
  cuUnpin(hostElems);
  free(hostElems);
}

/* Downloads a matrix' elements to host memory, freeing its device memory. */
void MatToHost(Mat m) {
  debug("downloading matrix from device");
  void* devElems = m->elems;
  MatNewElems(m, false);
  cuPin(m->elems, m->size);
  cuDownload(m->elems, devElems, m->size);
  cuUnpin(m->elems);
  cuFree(devElems);
}

/* Converts a 32-bit matrix to 64-bit. */
void MatPromote(Mat m) {
  struct Mat_ mOrig = *m;

  m->elemSize <<= 1; // double size
  m->size     <<= 1;
  m->pitch    <<= 1;
  MatNewElems(m, m->dev); // reallocate storage

  if (m->dev) {
    cuPromote(m->elems, mOrig.elems, mOrig.elemSize, mOrig.n2);
  } else {
    for (int64_t i = 0; i < mOrig.n2; i++) {
      switch (mOrig.elemSize) {
      case 4: ((double*)m->elems)[i] = ((float*)mOrig.elems)[i]; break;
      case 8: fatal("quad-precision not yet supported");         break;
      }
    }
  }

  MatFreeElems(&mOrig); // free the old elements
}

/* Returns a matrix element; ONLY for host matrices. */
inline double MatGet(Mat m, int row, int col) {
  union Elem e;
  memcpy(&e, elemAddr(m, row, col), m->elemSize);
  return ElemVal(&e, m->elemSize);
}

/* Sets a matrix element; ONLY for host matrices. */
inline void MatPut(Mat m, int row, int col, double elem) {
  union Elem e;
  ElemSet(&e, m->elemSize, elem);
  memcpy(elemAddr(m, row, col), &e, m->elemSize);
}
