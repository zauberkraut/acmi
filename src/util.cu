/* util.cu

   CUDA utilities. */

#include <cassert>

extern "C" {

size_t cuMemAvail() {
  size_t free, total;
  assert(cudaSuccess == cudaMemGetInfo(&free, &total));
  return free;
}

void* cuMalloc(size_t size) {
  void* p;
  assert(cudaSuccess == cudaMalloc(&p, size));
  return p;
}

void cuFree(void* p) { assert(cudaSuccess == cudaFree(p)); }

void cuClear(void* p, size_t size) {
  assert(cudaSuccess == cudaMemset(p, 0, size));
}

void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  assert(cudaSuccess == cudaMemcpy(devDst, hostSrc, size,
                                   cudaMemcpyHostToDevice));
}

void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  assert(cudaSuccess == cudaMemcpy(hostDst, devSrc, size,
         cudaMemcpyDeviceToHost));
}

void cuPin(void* p, size_t size) {
  assert(cudaSuccess == cudaHostRegister(p, size, cudaHostRegisterPortable));
}

void cuUnpin(void* p) {
  assert(cudaSuccess == cudaHostUnregister(p));
}

} // end extern "C"
