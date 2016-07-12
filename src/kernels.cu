/* kernels.cu

   Custom ACMI CUDA kernels. */

#include <assert.h>

extern "C" void* cuMalloc(size_t size) {
  void* p;
  assert(cudaMalloc(&p, size) == cudaSuccess);
  return p;
}

extern "C" void cuFree(void* p) { assert(cudaFree(p) == cudaSuccess); }

extern "C" void cuClear(void* p, size_t size) {
  assert(cudaMemset(p, 0, size) == cudaSuccess);
}

extern "C" void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  assert(cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice) ==
         cudaSuccess);
}

extern "C" void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  assert(cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
}
