# Reduction Case Study

Reduction is a common algorithmic operation used in parallel programming to
reduce an array of elements into a shorter array of elements or a single value.
This document exploits reduction to introduce some key considerations while
designing and optimizing GPU algorithms.

This repository hosts the sample code used in the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html).

## Structure

The coding style and the directory structure follow mostly that of
[rocPRIM](https://github.com/ROCm/rocPRIM), differing in a few ways:

- Unbound by the C++14 requirement of rocPRIM dictated by hipCUB and rocThrust,
  this repository uses C++20 as the baseline.
- As such, implementations are free to make use of some TMP/constexpr helper
  functions found within [`include/tmp_utils.hpp`](include/tmp_utils.hpp).
- The tests and benchmarks don't initialize resources multiple times, but do so
  just once and reuse the same input for tests/benchmarks of various sizes.
- Neither do tests, nor the benchmarks use prefixes for input initialization.
  Instead they both create a function object storing all states which tests
  capture by reference.
- "Diffing" the various implementations in succession reveals the minor changes
  between each version. `v0.hpp` is a simple Parallel STL implementation which
  is used for verification and a baseline of performance for comparison.
- The `example` folder holds the initial implementations of the various
  optimization levels of the benchmarks.
