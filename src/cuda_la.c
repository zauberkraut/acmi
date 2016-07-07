// cuda_la.c

#include <cuda_runtime.h>
#include "invmat.h"

static void check(cudaError_t err) {
  if (err != cudaSuccess) {
    fatal("%s:%d CUDA error %d: %s", err,
          cudaGetErrorString(err));
  }
}

uint64_t cuTotalMatBytes = 0;

void* cuMalloc(size_t size) {
  void* p;
  check(cudaMalloc(&p, size));
  return p;
}
void cuFree(void* p) {
  check(cudaFree(p));
}
void cuClear(void* p, size_t size) {
  check(cudaMemset(p, 0, size));
}
void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  check(cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice));
}
void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  check(cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost));
}

Mat cuNewMat(unsigned n) {
  Mat m = emptyMat(n);
  m->device = true;
  m->p = cuMalloc(m->size);
  cuTotalMatBytes += m->size;
  debug("allocated %.3lf MiB to %dx%d matrix on device; %.3lf MiB total",
        mibibytes(m->size), n, n, mibibytes(cuTotalMatBytes));
  return m;
}

void cuFreeMat(Mat m) {
  cuFree(m->p);
  m->p = 0;
  cuTotalMatBytes -= m->size;
  debug("freed %.3lf MiB from %dx%d matrix on device; %.3lf MiB remain",
        mibibytes(m->size), m->n, m->n, mibibytes(cuTotalMatBytes));
}

float cuElem(Mat m, int row, int col) {
  float e;
  cuDownload(&e, m->p + col*m->n + row, sizeof(e));
  return e;
}

void cuSetElem(Mat m, int row, int col, float e) {
  cuUpload(m->p + col*m->n + row, &e, sizeof(e));
}

void cuGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC) {
}

void cuGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC) {
}

float cuNorm(Mat mA) {
  return 0;
}
