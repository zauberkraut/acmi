// util.c

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static bool g_verbose = true;
void setVerbose(bool b) {
  g_verbose = b;
}

void debug(const char* msg, ...) {
  if (g_verbose) {
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    printf("\n");
    va_end(args);
  }
}

void warn(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  fprintf(stderr, "WARNING: ");
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
  va_end(args);
}

void fatal(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  fprintf(stderr, "FATAL: ");
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

double mibibytes(size_t size) {
  return (double)size/(1 << 20);
}
