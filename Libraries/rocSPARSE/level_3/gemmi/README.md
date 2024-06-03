# rocSPARSE Dense Matrix Sparse Matrix Multiplication Example

## Description

This example illustrates the use of the `rocsparse_gemmi` function, which performs a dense matrix-sparse matrix multiplication and scaling.

That is, it does the following operation:

$$
C =  \alpha \cdot op_a(A) \cdot op_b(B) + \beta \cdot C,
$$

where

- given a matrix $M$, $op_m(M)$ denotes one of the following:

  - $op_m(M) = M$ (identity)
  - $op_m(M) = M^T$ (transpose $M$: $M_{ij}^T = M_{ji}$)
  - $op_m(M) = M^H$ (conjugate transpose/Hermitian $M$: $M_{ij}^H = \bar M_{ji}$),

- $A$ is a dense matrix $m \times k$,
- $B$ is a sparse matrix of size $k \times n$,
- $\alpha$ and $\beta$ are scalars,
- $C$ is a dense matrix of size $m \times n$.

### Application flow

1. Set up input data.
2. Allocate device memory and copy input data to device.
3. Prepare for rocSPARSE function call by creating a handle and matrix descriptor.
4. Perform the computation.
5. Copy result from device.
6. Free rocSPARSE resources and device memory.

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

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_pointer_mode` controls whether scalar parameters must be allocated on the host (`rocsparse_pointer_mode_host`) or on the device (`rocsparse_pointer_mode_device`). It is controlled by `rocsparse_set_pointer_mode`.
- `rocsparse_operation trans`: matrix operation applied to the given matrix. The following values are accepted:
  - `rocsparse_operation_none`: identity operation $op(M) = M$.
  - `rocsparse_operation_transpose`: transpose operation $op(M) = M^\mathrm{T}$.
  - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $op(M) = M^\mathrm{H}$.
  Currently, operation on $A$ must be `rocsparse_operation_none` and operation on $B$ must be `rocsparse_operation_transpose`. The other options are not yet supported.
- `rocsparse_mat_descr descr`: holds all properties of a matrix.
- `rocsparse_[sdcz]gemmi` performs the operation $C =  \alpha \cdot op_a(A) \cdot op_b(B) + \beta \cdot C$ for $C$. The correct function signature should be chosen based on the datatype of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_dgemmi`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_mat_descr`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_operation_transpose`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
