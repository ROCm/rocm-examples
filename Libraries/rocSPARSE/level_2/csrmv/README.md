# rocSPARSE Level 2 CSR Matrix-Vector Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication using CSR storage format.

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot op(A) \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $op(A)$ is a sparse matrix in CSR format with `rocsparse_operation`, which is described below in [Key APIs and Concepts - rocSPARSE](#rocsparse) in more detail.

## Application flow

1. Set up a sparse matrix in CSR format. Allocate an x and a y vector and set up $\alpha$ and $\beta$ scalars.
2. Set up handle, matrix descriptor and matrix info variables.
3. Allocate device memory and copy input matrix and vectors from host to device.
4. Compute a sparse matrix multiplication, using CSR (compressed sparse row) storage format.
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts

### CSR Matrix Storage Format

The [Compressed Sparse Row (CSR) storage format](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/basics.html#csr-storage-format) describes an $m \times n$ sparse matrix with three arrays.

Defining

- `m`: number of rows
- `n`: number of columns
- `nnz`: number of non-zero elements

we can describe a sparse matrix using the following arrays:

- `csr_val`: array storing the non-zero elements of the matrix.
- `csr_row_ptr`: given $i \in [0, m]$
  - if $` 0 \leq i < m `$, `csr_row_ptr[i]` stores the index of the first non-zero element in row $i$ of the matrix
  - if $i = m$, `csr_row_ptr[i]` stores `nnz`.

  This way, row $j \in [0, m)$ contains the non-zero elements of indices from `csr_row_ptr[j]` to `csr_row_ptr[j+1]-1`. Therefore, the corresponding values in `csr_val` can be accessed from `csr_row_ptr[j]` to `csr_row_ptr[j+1]-1`.
- `csr_col_ind`: given $i \in [0, nnz-1]$, `csr_col_ind[i]` stores the column of the $i^{th}$ non-zero element in the matrix.

The CSR matrix is sorted by column indices in the same row, and each pair of indices appear only once.

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

Therefore, the CSR representation of $A$ is:

```cpp
m = 3

n = 5

nnz = 8

csr_val = { 1, 2, 3, 4, 5, 6, 7, 8 }

csr_row_ptr = { 0, 3, 5, 8 }

csr_col_ind = { 0, 1, 3, 1, 2, 0, 3, 4 }
```

### rocSPARSE

- `rocsparse_[dscz]csrmv(...)` is the solver with four different function signatures depending on the type of the input matrix:
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
- `rocsparse_create_mat_info`
- `rocsparse_dcsrmv`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_destroy_mat_info`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_mat_descr`
- `rocsparse_mat_info`
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
