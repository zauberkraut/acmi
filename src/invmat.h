// invmat.h

#ifndef INVMAT_H
#define INVMAT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int MAX_MAT_DIM = 32768;

typedef struct {
  const char* path;
  bool sparse;
  bool binary;
  bool symmetric;
  bool skew;
  int n;
  int nEntries; // valid only if sparse
  int nNonzero; // ''
  double sparsity;
  int64_t size;
} MatInfo;

typedef struct {
  int n;
  int64_t n2;
  int64_t size;
  float* p;
  bool dev;
} Mat_;
typedef Mat_* Mat;

typedef enum {BLAS_IMPL, CUBLAS_IMPL, LU_IMPL} Impl;

static inline double mibibytes(uint64_t size) {
  return (double)size/(1 << 20);
}

void setVerbose(bool b);
void debug(const char* msg, ...);
void warn(const char* msg, ...);
void fatal(const char* msg, ...);
void writeMat(FILE* fout, Mat m);
void saveMat(const char* path, Mat m);
MatInfo readMatInfo(const char* path);
void loadMat(Mat mA, MatInfo info);
void genRandIntMat(Mat mA, const char* outPath);
void genRandRealMat(Mat mA, const char* outPath);

Mat hostNewMat(int n);
Mat devNewMat(int n);
void clearMat(Mat m);
void freeMat(Mat m);
float elem(Mat m, int row, int col);
void setElem(Mat m, int row, int col, float e);

extern void (*gemm)(float alpha, Mat mA, Mat mB, float beta, Mat mC);
extern void (*geam)(float alpha, Mat mA, float beta, Mat mB, Mat mC);
extern float (*norm)(Mat mA);
float invert(Mat mA, Mat mR, float maxError, int maxStep, bool quadConv);

void blasGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void blasGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float blasNorm(Mat mA);

void* cuMalloc(size_t size);
void cuFree(void* p);
void cuClear(void* p, size_t size);
void cuUpload(void* devDst, const void* hostSrc, size_t size);
void cuDownload(void* hostDst, const void* devSrc, size_t size);
void cublInit(int n);
void cublShutDown();
void cublGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void cublGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float cublNorm(Mat mA);
float luInvert(Mat mA, Mat mR);

#endif
