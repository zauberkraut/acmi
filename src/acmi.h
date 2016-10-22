/* acmi.h

   Internal header for the ACMI project. */

#ifndef ACMI_H
#define ACMI_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { MAX_MAT_DIM = 32768, MAX_ELEM_SIZE = 8 };

struct Mat_;
typedef struct Mat_* Mat;
/* For generic handling of floating-point precisions. */
union Elem {
  float fp32;
  double fp64;
};

// util.c
void debug(const char* msg, ...);
void warn(const char* msg, ...);
void fatal(const char* msg, ...);
int iMin(int a, int b);
double mibibytes(size_t size);
bool checkDevMemEnough(int n, int elemSize, int matCount);
Mat MatLoad(const char* path, int elemSize, int matCount);
Mat MatNewRand(int n, int elemSize, double maxElem, bool symm, bool real,
               bool neg, bool diagDom, bool useHardwareRNG);
void MatWrite(Mat m, const char* path);
void MatPrint(Mat m);

// mat.c
Mat MatNew(int n, int elemSize, bool dev);
Mat MatBuild(Mat m);
void MatFree(Mat m);
void MatClear(Mat m);
int MatN(Mat m);
int64_t MatN2(Mat m);
int MatElemSize(Mat m);
size_t MatSize(Mat m);
size_t MatPitch(Mat m);
void* MatElems(Mat m);
bool MatDev(Mat m);
void MatToDev(Mat m);
void MatToHost(Mat m);
void MatPromote(Mat m);
double MatGet(Mat m, int row, int col);
void MatPut(Mat m, int row, int col, double elem);

// invert.c
double altmanInvert(const Mat mA, Mat* const mRp, const int convOrder,
                    const double errLimit, const int msLimit,
                    double convRateLimit, bool safeR0);

// linalg.c
void gpuSetUp(const int maxBlocksPerKernel, const int n);
void gpuShutDown();
void transpose(double alpha, Mat mA, Mat mT);
void gemm(double alpha, Mat mA, Mat mB, double beta, Mat mC);
void geam(double alpha, Mat mA, double beta, Mat mB, Mat mC);
void addId(double alpha, Mat mA);
double nrm2(Mat mA);
double trace(Mat mA);
double minusIdNrm2(Mat mA);

// util.cu
#define CUCHECK cuCheck(__FILE__, __LINE__)
void cuCheck(const char* fname, const size_t lnum);
size_t cuMemAvail();
void* cuMalloc(size_t size);
void cuFree(void* p);
void cuClear(void* p, size_t size);
void cuUpload(void* devDst, const void* hostSrc, size_t size);
void cuDownload(void* hostDst, const void* devSrc, size_t size);
void cuPin(void* p, size_t size);
void cuUnpin(void* p);

// kernels.cu
void cuSetUp(const int maxBlocksPerKernel, const int n);
void cuShutDown();
void cuPromote(void* dst, void* src, int srcElemSize, int64_t n2);
void cuAddId(double alpha, void* elems, int n, int elemSize);
double cuTrace(void* elems, int n, int elemSize);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
