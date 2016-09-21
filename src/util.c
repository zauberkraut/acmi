/* util.c

   ACMI utility functions. */

#include <stdarg.h>
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
  exit(1);
}

double mibibytes(size_t size) {
  return (double)size/(1 << 20);
}
