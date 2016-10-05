/* util.cu

   CUDA utilities. */

#include "acmi.h"

extern "C" {

void cuCheck(const char* fname, const size_t lnum) {
  auto r = cudaPeekAtLastError();
  if (cudaSuccess != r) {
    fatal("%s line %d: %s", fname, lnum, cudaGetErrorString(r));
  }
}

size_t cuMemAvail() {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  CUCHECK;
  return free;
}

void* cuMalloc(size_t size) {
  void* p;
  cudaMalloc(&p, size);
  CUCHECK;
  return p;
}

void cuFree(void* p) {
  cudaFree(p);
  CUCHECK;
}

void cuClear(void* p, size_t size) {
  cudaMemset(p, 0, size);
  CUCHECK;
}

void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice);
  CUCHECK;
}

void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost);
  CUCHECK;
}

void cuPin(void* p, size_t size) {
  cudaHostRegister(p, size, cudaHostRegisterPortable);
  CUCHECK;
}

void cuUnpin(void* p) {
  cudaHostUnregister(p);
  CUCHECK;
}

} // end extern "C"
