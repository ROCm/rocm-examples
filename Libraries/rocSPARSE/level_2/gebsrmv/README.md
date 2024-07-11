# rocSPARSE Level 2 GEBSR Matrix-Vector Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication using GEBSR storage format.

The operation calculates the following product:

$$\hat{y} = \alpha \cdot op(A) \cdot x + \beta \cdot y$$

where

- $\alpha$ and $\beta$ are scalars,
- $x$ and $y$ are dense vectors,
- $A$ is an $m\times n$ sparse matrix, and
- $op(A)$ is one of the following:
  - $op(A) = A$ (identity)
  - $op(A) = A^T$ (transpose $A$: $A_{ij}^T = A_{ji}$)
  - $op(A) = A^H$ (conjugate transpose/Hermitian $A$: $A_{ij}^H = \bar A_{ji}$).

## Application flow

1. Setup input data.
2. Allocate device memory and offload input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare utility variables for rocSPARSE gebsrmv invocation.
5. Call gebsrmv to perform $y = \alpha * A * x + beta * y$. <!-- markdownlint-disable-line no-space-in-emphasis -->
6. Copy solution to host from device.
7. Clear rocSPARSE allocations on device and device arrays.
8. Print results to standard output.

## Key APIs and Concepts

### GEBSR Matrix Storage Format

The [General Block Compressed Sparse Row (GEBSR) storage format](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/basics.html#gebsr-storage-format) describes a sparse matrix using three arrays. The idea behind this storage format is the same as for the BSR format, but the blocks in which the sparse matrix is split are not squared. All of them are of `bsr_row_dim` $\times$ `bsr_col_dim` size.

Therefore, defining

- `mb`: number of rows of blocks
- `nb`: number of columns of blocks
- `nnzb`: number of non-zero blocks
- `bsr_row_dim`: number of rows in each block
- `bsr_col_dim`: number of columns in each block

we can describe a sparse matrix using the following arrays:

- `bsr_val`: contains the elements of the non-zero blocks of the sparse matrix. The elements are stored block by block in column- or row-major order. That is, it is an array of size `nnzb` $\cdot$ `bsr_row_dim` $\cdot$ `bsr_col_dim`.

- `bsr_row_ptr`: given $i \in [0, mb]$
  - if $` 0 \leq i < mb `$, `bsr_row_ptr[i]` stores the index of the first non-zero block in row $i$ of the block matrix
  - if $i = mb$, `bsr_row_ptr[i]` stores `nnzb`.

  This way, row $j \in [0, mb)$ contains the non-zero blocks of indices from `bsr_row_ptr[j]` to `bsr_row_ptr[j+1]-1`. The corresponding values in `bsr_val` can be accessed from `bsr_row_ptr[j] * bsr_row_dim * bsr_col_dim` to `(bsr_row_ptr[j+1]-1) * bsr_row_dim * bsr_col_dim`.

- `bsr_col_ind`: given $i \in [0, nnzb-1]$, `bsr_col_ind[i]` stores the column of the $i^{th}$ non-zero block in the block matrix.

Note that, for a given $m\times n$ matrix, if $m$ is not evenly divisible by the row block dimension or $n$ is not evenly divisible by the column block dimension then zeros are padded to the matrix so that $mb$ and $nb$ are the smallest integers greater than or equal to $`\frac{m}{\texttt{bsr\_row\_dim}}`$ and $`\frac{n}{\texttt{bsr\_col\_dim}}`$, respectively.

### rocSPARSE

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.

- `rocsparse_[dscz]gebsrmv(...)` performs the sparse matrix-dense vector multiplication $\hat{y}=\alpha \cdot op(A) x + \beta \cdot y$ using the GEBSR format. The correct function signature should be chosen based on the datatype of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_operation trans`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = M^\mathrm{H}$

  Currently, only `rocsparse_operation_none` is supported for `rocsparse_[dscz]gebsrmv`.

- `rocsparse_mat_descr`: descriptor of the sparse GEBSR matrix.

- `rocsparse_direction` block storage major direction with the following options:
  - `rocsparse_direction_column`
  - `rocsparse_direction_row`

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_dgebsrmv`
- `rocsparse_direction`
- `rocsparse_direction_row`
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
