# rocSPARSE Level 2 COO Matrix-Vector Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication using COO storage format.

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot op(A) \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $op(A)$ is a sparse matrix in COO format, result of applying one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

## Application flow

1. Set up a sparse matrix in COO format. Allocate an x and a y vector and set up $\alpha$ and $\beta$ scalars.
2. Set up a handle, a matrix descriptor and a matrix info.
3. Allocate device memory and copy input matrix and vectors from host to device.
4. Compute a sparse matrix multiplication, using COO storage format.
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts

### COO Matrix Storage Format

The coordinate (COO) storage format represents an $m \times n$ matrix by

- `m`: number of rows
- `n`: number of columns
- `nnz`: number of non-zero elements
- `coo_val`: array of non-zero elements
- `coo_row_ind`: array of row indices of each elements in `coo_val`
- `coo_col_ind`: array of column indices of each elements in `coo_val`

The COO matrix is sorted by row indices, and by column indices in the same row.

### rocSPARSE

- `rocsparse_[dscz]coomv(...)` is the solver with four different function signatures depending on the type of the input matrix:
  - `d` double-precision real (`double`)
  - `s` single-precision real (`float`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_operation`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = M^\mathrm{H}$

- `rocsparse_mat_descr`: holds all properties of a matrix.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_dcoomv`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_mat_descr`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
