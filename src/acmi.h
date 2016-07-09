// acmi.h

#ifndef ACMI_H
#define ACMI_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int MAX_MAT_DIM = 32768;

struct Mat_;
typedef struct Mat_* Mat;
typedef enum {BLAS_IMPL, CUBLAS_IMPL, LU_IMPL} Impl;

// util.c
void setVerbose(bool b);
void debug(const char* msg, ...);
void warn(const char* msg, ...);
void fatal(const char* msg, ...);
double mibibytes(size_t size);

// mat.c
Mat MatNew(int n, bool dev);
void MatFree(Mat m);
void MatClear(Mat m);
int MatN(Mat m);
int64_t MatN2(Mat m);
size_t MatSize(Mat m);
float* MatElements(Mat m);
bool MatDev(Mat m);
double MatTrace(Mat m);
void MatToDev(Mat m);
void MatToHost(Mat m);
float MatGet(Mat m, int row, int col);
void MatPut(Mat m, int row, int col, float e);
Mat MatLoad(const char* path, bool attrOnly);
void MatWrite(Mat m, const char* path);
Mat MatRandDiagDom(int n, bool real, bool symmetric);

// la.c
float altmanInvert(Mat mA, Mat mR, float maxError, int maxStep, bool quadConv);

// blas.c
void blasGemm(float alpha, Mat mA, Mat mB, float beta, Mat mC);
void blasGeam(float alpha, Mat mA, float beta, Mat mB, Mat mC);
float blasNorm(Mat mA);

// cuda.c
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
