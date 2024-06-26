# rocSPARSE Level 2 ELL Matrix-Vector Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication using ELL storage format.

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot op(A) \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $op(A)$ is a sparse matrix in ELL format, result of applying one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

## Application flow

1. Set up a sparse matrix in ELL format. Allocate an x and a y vector and set up $\alpha$ and $\beta$ scalars.
2. Set up a handle, a matrix descriptor and a matrix info.
3. Allocate device memory and copy input matrix and vectors from host to device.
4. Compute a sparse matrix multiplication, using ELL storage format.
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts

### ELL Matrix Storage Format

The Ellpack-Itpack (ELL) storage format represents an $m \times n$ matrix in column-major layout, by fixing the number of non-zeros in each column. A matrix is stored using the following arrays:

- `m`: number of rows
- `n`: number of columns
- `nnz`: number of non-zero elements
- `ell_width`: maximum number of non-zero elements per row
- `ell_val`: array of data with $`m \times \texttt{ell\_with}`$ elements
- `ell_col_ind`: array of column indices with $`m \times \texttt{ell\_with}`$ elements. Rows with less than `ell_width` non-zero elements are padded:
  - `ell_val` with zeroes
  - `ell_col_ind` with $-1$

For instance, consider a sparse matrix as

$$
A=
\left(
\begin{array}{ccccc}
1 & 2 & 0 & 3 & 0 \\
0 & 4 & 5 & 0 & 0 \\
6 & 0 & 0 & 7 & 8
\end{array}
\right)
$$

Therefore, the ELL representation of $A$ is:

```cpp
m = 3

n = 5

ell_width = 3

ell_val = { 1, 4, 6,
            2, 5, 7,
            3, 0, 8 }

ell_col_ind = { 0, 1, 0,
                1, 2, 3,
                3, -1, 4 }
```

### rocSPARSE

- `rocsparse_[dscz]ellmv(...)` is the solver with four different function signatures depending on the type of the input matrix:
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
- `rocsparse_dellmv`
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
