/* test.c

   ACMI unit tests. */

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include "acmi.h"

/* Tests host matrix creation and access. */
static void testMatBasic(void** state) {
  const int n = 1024;

  Mat m = MatNew(n, 4, false);
  assert_int_equal(MatN(m), n);
  assert_int_equal(MatN2(m), n*n);
  assert_int_equal(MatElemSize(m), 4);
  assert_int_equal(MatSize(m), n*n*4);
  assert_int_equal(MatPitch(m), n*4);
  assert_non_null(MatElems(m));
  assert_false(MatDev(m));

  MatClear(m);
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < n; row++) {
      assert_true(0. == MatGet(m, row, col));
    }
  }

  MatPut(m, 0, 0, 1);
  MatPut(m, n - 1, n - 1, 1);
  MatPut(m, n/2, n/2, 1);
  MatPut(m, n/3, n/4, 1);
  MatPut(m, n/4, n/3, 1);
  assert_true(1. == MatGet(m, 0, 0));
  assert_true(1. == MatGet(m, n - 1, n - 1));
  assert_true(1. == MatGet(m, n/2, n/2));
  assert_true(1. == MatGet(m, n/3, n/4));
  assert_true(1. == MatGet(m, n/4, n/3));

  MatPut(m, 0, 0, 0);
  MatPut(m, n/3, n/4, 0);
  assert_true(0. == MatGet(m, 0, 0));
  assert_true(0. == MatGet(m, n/3, n/4));

  MatFree(m);
}

/* Basic double-precision matrix tests. */
static void testMat64(void** state) {
  const int n = 1024;

  Mat m = MatNew(n, 8, false);
  assert_int_equal(MatN(m), n);
  assert_int_equal(MatN2(m), n*n);
  assert_true(8 == MatElemSize(m));
  assert_int_equal(MatElemSize(m), 8);
  assert_int_equal(MatSize(m), n*n*8);
  assert_int_equal(MatPitch(m), n*8);
  assert_non_null(MatElems(m));
  assert_false(MatDev(m));

  MatPut(m, 0, 0, 1);
  MatPut(m, n - 1, n - 1, 1);
  MatPut(m, n/2, n/2, 1);
  MatPut(m, n/3, n/4, 1);
  MatPut(m, n/4, n/3, 1);
  assert_true(1. == MatGet(m, 0, 0));
  assert_true(1. == MatGet(m, n - 1, n - 1));
  assert_true(1. == MatGet(m, n/2, n/2));
  assert_true(1. == MatGet(m, n/3, n/4));
  assert_true(1. == MatGet(m, n/4, n/3));

  MatFree(m);
}

/* Tests the symmetry of a randomly-generated, symmetric matrix. */
static void testMatNewRandSymm(void** state) {
  const int n = 1024;

  Mat m = MatNewRand(n, 4, 16, true, false, true, false, false);

  for (int row = 0; row < n; row++) {
    for (int col = row + 1; col < n; col++) {
      assert_true(MatGet(m, row, col) == MatGet(m, col, row));
    }
  }
}

int main() {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(testMatBasic),
    cmocka_unit_test(testMat64),
    cmocka_unit_test(testMatNewRandSymm),
  };

  return cmocka_run_group_tests(tests, 0, 0);
}
