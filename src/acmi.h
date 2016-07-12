// acmi.h

#ifndef ACMI_H
#define ACMI_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int MAX_MAT_DIM = 32768;
typedef enum {CPU_IMPL, CUBLAS_IMPL, LU_IMPL} Impl;

struct Mat_;
typedef struct Mat_* Mat;

// util.c
void setVerbose(bool b);
void debug(const char* msg, ...);
void warn(const char* msg, ...);
void fatal(const char* msg, ...);
double mibibytes(size_t size);

// mat.c
Mat MatNew(int n, bool doublePrec, bool dev);
Mat MatBuild(Mat m);
void MatFree(Mat m);
void MatClear(Mat m);
int MatN(Mat m);
int64_t MatN2(Mat m);
bool MatDouble(Mat m);
int MatElemSize(Mat m);
size_t MatSize(Mat m);
size_t MatPitch(Mat m);
void* MatElems(Mat m);
void* MatCol(Mat m, int col);
bool MatDev(Mat m);
bool MatSymm(Mat m);
bool MattSparse(Mat m);
double MatTrace(Mat m);
void MatToDev(Mat m);
void MatToHost(Mat m);
double MatGet(Mat m, int row, int col);
void MatPut(Mat m, int row, int col, double elem);
Mat MatLoad(const char* path, bool doublePrec, bool attrOnly);
void MatWrite(Mat m, const char* path);
Mat MatRandDiagDom(int n, bool doublePrec, bool symm);
void MatDebug(Mat m);

// invert.c
double altmanInvert(Mat mA, Mat mR, double maxError, int maxStep,
                    bool quadConv);

// blas.c
void initCublas();
void shutDownCublas();
void gemm(double alpha, Mat mA, Mat mB, double beta, Mat mC);
void gemmT(double alpha, Mat mA, Mat mB, double beta, Mat mC);
void geam(double alpha, Mat mA, double beta, Mat mB, Mat mC);
double norm(Mat mA);
double luInvert(Mat mA, Mat mR);

// kernels.c
void* cuMalloc(size_t size);
void cuFree(void* p);
void cuClear(void* p, size_t size);
void cuUpload(void* devDst, const void* hostSrc, size_t size);
void cuDownload(void* hostDst, const void* devSrc, size_t size);

#endif
