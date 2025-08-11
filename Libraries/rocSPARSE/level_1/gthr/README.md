# rocSPARSE Level 1 Gather elements from a dense vector and stores them into a sparse vector

## Description

This example illustrates the use of the `rocSPARSE` level 1 routine which gathers elements from a dense vector and store them into a sparse vector.

## Application flow

1. Allocate a sparse x vector and a dense y vector.
2. Set up a handle.
3. Allocate device memory and copy input vectors from host to device.
4. Computing gather operation
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts

### rocSPARSE

- `rocsparse_[dscz]gthr(...)` accepts four different function signatures depending on the data type of the input sparse and dense vectors:
  - `d` double-precision real (`double`)
  - `s` single-precision real (`float`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_index_base`: index base type with the following options:
  - `rocsparse_index_base_zero`: the sparse vector $x$ has zero index base
  - `rocsparse_index_base_one`: the sparse vector $x$ has one index base

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_sgthr`
- `rocsparse_destroy_handle`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
