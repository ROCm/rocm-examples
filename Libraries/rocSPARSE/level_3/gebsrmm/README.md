# rocSPARSE Level-3 GEBSR Matrix-Matrix Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 3 sparse matrix-matrix multiplication using GEBSR storage format.

The operation calculates the following product:

$\hat{C} = \alpha \cdot op_a(A) \cdot op_b(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a sparse matrix in GEBSR format
- $B$ and $C$ are dense matrices
- and $op_a(A)$ and $op_b(B)$ are the result of applying to matrices $A$ and $B$, respectively, one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

## Application flow

1. Set up a sparse matrix in GEBSR format. Allocate an $A$ and a $B$ matrix and set up $\alpha$ and $\beta$ scalars.
2. Set up a handle, a matrix descriptor.
3. Allocate device memory and copy input matrices from host to device.
4. Compute a sparse matrix multiplication, using GEBSR storage format.
5. Copy the result matrix from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

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

- `rocsparse_[sdcz]gebsrmm(...)` is the matrix-matrix multiplication solver with four different function signatures depending on the type of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_operation`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = A^\mathrm{H}$

  Currently, only `rocsparse_operation_none` is supported.
- `rocsparse_mat_descr`: descriptor of the sparse BSR matrix.

- `rocsparse_direction` block storage major direction with the following options:
  - `rocsparse_direction_column`
  - `rocsparse_direction_row`

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_dgebsrmm`
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
