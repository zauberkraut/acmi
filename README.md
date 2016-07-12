# ACMI
======
***A***CMI's a ***C***onvergent ***M***atrix ***I***nverter.

ACMI implements M. Altman's cubically-convergent, iterative algorithm to compute
matrix inverses. GPU-hosted computation is currently supported using CUDA.

---

Dependencies:
  * GCC
  * OpenBLAS
  * The CUDA SDK

---

To build:  
`make`

To build with debugging symbols:  
`make debug`

For usage instructions, run (after building):  
`build/acmi`

To invert a 10000x10000 random, diagonally-dominant integer matrix using cuBLAS
without writing any matrix files, run:  
`build/acmi ?10000`

Large, sparse matrices can be found [here](
https://www.cise.ufl.edu/research/sparse/matrices/
"The UF Sparse Matrix Collection"
).

---

TODO:
  * Unit tests!
  * Return errors instead of calling fatal()
  * OpenCL support
  * Library build
  * Fix makefile quirks
  * Sprout CLI handler out of main() and handle conflicting options
  * Portable PRNG seeding
  * Time-limited inversion
