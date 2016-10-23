# ACMI
***A***CMI's a ***C***onvergent ***M***atrix ***I***nverter.

ACMI implements M. Altman's cubically-convergent, iterative algorithm to compute
matrix inverses. GPU-hosted computation is currently supported using CUDA. ACMI
is being developed on the Pascal architecture.

---

Dependencies:
  * CUDA SDK 8.0
  * GCC
  * GNU Make
  * OpenBLAS
  * CMocka for unit tests

---

To build:  
`make`

To build with debugging symbols:  
`make debug`

To run unit tests:  
`make test`

For usage instructions, run (after building):  
`acmi`

To invert a 10000x10000 random, diagonally-dominant integer matrix using CUDA
without writing any matrix files, run:  
`acmi @10000 -D`

Large, sparse matrices can be found [here](
https://www.cise.ufl.edu/research/sparse/matrices/
"The UF Sparse Matrix Collection"
).

This project uses Matrix Market I/O routines from:  
http://math.nist.gov/MatrixMarket/mmio-c.html

---

TODO:
  * More unit tests!
  * Make reported iteration count consistent
  * OpenCL/Vulkan support
  * Library build
  * Fix makefile quirks
  * Sprout CLI handler out of main() and handle conflicting options
  * Portable PRNG seeding
  * Optimized multiplication of symmetric or sparse matrices during the first iteration
  * Double double support
  * Increase output file precision
