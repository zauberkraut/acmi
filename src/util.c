/* util.c

   ACMI utility functions. */

#include <cpuid.h>
#include <immintrin.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void print(FILE* f, const char* s, va_list args) {
  vfprintf(f, s, args);
  fprintf(f, "\n");
  va_end(args);
}

void debug(const char* s, ...) {
  va_list args; va_start(args, s); print(stdout, s, args);
}

void warn(const char* s, ...) {
  fprintf(stderr, "WARNING: ");
  va_list args; va_start(args, s);
  print(stderr, s, args);
}

void fatal(const char* s, ...) {
  fprintf(stderr, "FATAL: ");
  va_list args; va_start(args, s);
  print(stderr, s, args);
  abort();
}

double mibibytes(size_t size) {
  return (double)size/(1 << 20);
}

bool f16cSupported() {
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ecx & bit_F16C;
}

// 0 => round to nearest even
uint16_t singleToHalf(float f)    { return _cvtss_sh(f, 0); }
float    halfToSingle(uint16_t h) { return _cvtsh_ss(h); }
