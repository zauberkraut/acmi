// invmat.h

#ifndef INVMAT_H
#define INVMAT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int MAX_MAT_DIM = 20000;

typedef struct {
  const char* path;
  bool sparse;
  bool binary;
  bool symmetric;
  bool skew;
  unsigned n;
  unsigned nEntries; // valid only if sparse
  unsigned nNonzero; // ''
  double sparsity;
  uint64_t size;
} MatInfo;

typedef struct {
  unsigned n;
  uint64_t n2;
  uint64_t size;
  float* p;
  bool device;
} Mat_;
typedef Mat_* Mat;

typedef enum {CPU_IMPL, CUBLAS_IMPL, CUDA_IMPL, LU_IMPL} Impl;

static inline double mibibytes(uint64_t size) {
  return (double)size/(1 << 20);
}

extern uint64_t cpuTotalMatBytes;
extern uint64_t cuTotalMatBytes;

void setVerbose(bool b);
void debug(const char* msg, ...);
void warn(const char* msg, ...);
void fatal(const char* msg, ...);
void writeMat(FILE* fout, Mat m);
MatInfo readMatInfo(const char* path);
Mat loadMat(MatInfo info);
Mat randIntMat(unsigned n);
Mat randRealMat(unsigned n);

extern Mat (*newMat)(unsigned n);
Mat emptyMat(unsigned n);
Mat zeroMat(unsigned n);
void clearMat(Mat m);
void freeMat(Mat m);
void bound(Mat m, int row, int col);
float (*elem)(Mat m, int row, int col);
void (*setElem)(Mat m, int row, int col, float e);
void init(unsigned n, Impl impl, bool quadConv);
void shutDown();
float measureAR(Mat mA, Mat mR);
float invert(Mat mA, Mat mR, float maxError, int maxStep);

Mat cpuNewMat(unsigned n);
void cpuFreeMat(Mat m);
Mat cpuCopyMat(Mat m);
float cpuElem(Mat m, int row, int col);
void cpuSetElem(Mat m, int row, int col, float e);
void cpuGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void cpuGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float cpuNorm(Mat mA);

void cublInit(unsigned n);
void cublShutDown();
void cublGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void cublGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float cublNorm(Mat mA);
float luInvert(Mat mA, Mat mR);

void* cuMalloc(size_t size);
void cuFree(void* p);
void cuClear(void* p, size_t size);
void cuUpload(void* devDst, const void* hostSrc, size_t size);
void cuDownload(void* hostDst, const void* devSrc, size_t size);
Mat cuNewMat(unsigned n);
void cuFreeMat(Mat m);
Mat cuCopyMat(Mat m);
float cuElem(Mat m, int row, int col);
void cuSetElem(Mat m, int row, int col, float e);
void cuGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void cuGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float cuNorm(Mat mA);

#endif
