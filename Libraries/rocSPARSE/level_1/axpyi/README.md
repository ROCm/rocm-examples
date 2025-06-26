# rocSPARSE Level 1 Scale a sparse vector and add it to a dense vector

## Description

This example illustrates the use of the `rocSPARSE` level 1 routine which scales a sparse vector and adds it to a dense vector:

$$\mathbf{y} = \mathbf{y} + \alpha \cdot \mathbf{x}$$

where

- $\alpha$ is a scalar
- $\mathbf{y}$ is a dense vector
- $\mathbf{x}$ is a sparse vector

## Application flow

1. Allocate a sparse x vector and a dense y vector and set up $\alpha$ scalar.
2. Set up a handle.
3. Allocate device memory and copy input vectors from host to device.
4. Computing scaling and addition of vectors
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts

### rocSPARSE

- `rocsparse_[dscz]axpyi(...)` accepts four different function signatures depending on the data type of the input sparse and dense vectors:
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
- `rocsparse_saxpyi`
- `rocsparse_destroy_handle`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
