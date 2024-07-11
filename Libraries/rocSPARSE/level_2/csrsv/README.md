# rocSPARSE Level 2 CSR Triangular Solver Example

## Description

This example illustrates the use of the `rocSPARSE` level 2 triangular solver using the CSR storage format.

This triangular solver is used to solve a linear system of the form

$$
op(A) \cdot y = \alpha \cdot x,
$$

where

- $A$ is a sparse triangular matrix of order $n$ whose elements are the coefficients of the equations,
- $op(A)$ is one of the following:
  - $op(A) = A$ (identity)
  - $op(A) = A^T$ (transpose $A$: $A_{ij}^T = A_{ji}$)
  - $op(A) = A^H$ (conjugate transpose/Hermitian $A$: $A_{ij}^H = \bar A_{ji}$),
- $\alpha$ is a scalar,
- $x$ is a dense vector of size $m$ containing the constant terms of the equations, and
- $y$ is a dense vector of size $n$ which contains the unknowns of the system.

Obtaining solution for such a system consists on finding concrete values of all the unknowns such that the above equality holds.

### Application flow

1. Setup input data.
2. Allocate device memory and offload input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare utility variables for rocSPARSE csrsv invocation.
5. Perform analysis step.
6. Perform triangular solve $op(A) \cdot y = \alpha \cdot x$.
7. Check results obtained. If no zero-pivots, copy solution vector $y$ from device to host and compare with expected result.
8. Free rocSPARSE resources and device memory.
9. Print validation result.

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

- `rocsparse_operation trans`: matrix operation applied to the given input matrix. The following values are accepted:
  - `rocsparse_operation_none`: identity operation $op(M) = M$.
  - `rocsparse_operation_transpose`: transpose operation $op(M) = M^\mathrm{T}$.
  - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $op(M) = M^\mathrm{H}$. This operation is not yet supported for `rocsparse_[sdcz]_csrsv_solve`.

- `rocsparse_mat_descr descr`: holds all properties of a matrix. The properties set in this example are the following:
  - `rocsparse_diag_type`: indicates whether the diagonal entries of a matrix are unit elements (`rocsparse_diag_type_unit`) or not (`rocsparse_diag_type_non_unit`).
  - `rocsparse_fill_mode`: indicates whether a (triangular) matrix is lower (`rocsparse_fill_mode_lower`) or upper (`rocsparse_fill_mode_upper`) triangular.

- `rocsparse_[sdcz]csrsv_buffer_size` allows to obtain the size (in bytes) of the temporary storage buffer required for the `rocsparse_[sdcz]csrsv_analysis` and `rocsparse_[sdcz]csrsv_solve` functions. The character matched in `[sdcz]` coincides with the one matched in any of the mentioned functions.

- `rocsparse_solve_policy policy`: specifies the policy to follow for triangular solvers and factorizations. The only value accepted is `rocsparse_solve_policy_auto`.

- `rocsparse_[sdcz]csrsv_solve` solves a sparse triangular linear system $op(A) \cdot y = \alpha \cdot x$. The correct function signature should be chosen based on the datatype of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_analysis_policy analysis`: specifies the policy to follow for analysis data. The following values are accepted:
  - `rocsparse_analysis_policy_reuse`: the analysis data gathered is re-used.
  - `rocsparse_analysis_policy_force`: the analysis data will be re-built.

- `rocsparse_[sdcz]csrsv_analysis` performs the analysis step for `rocsparse_[sdcz]csrsv_solve`. The character matched in `[sdcz]` coincides with the one matched in `rocsparse_[sdcz]csrsv_solve`.

- `rocsparse_csrsv_zero_pivot(rocsparse_handle, rocsparse_mat_info, rocsparse_int *position)` returns `rocsparse_status_zero_pivot` if either a structural or numerical zero has been found during the execution of `rocsparse_[sbcz]csrsv_solve(....)` and stores in `position` the index $i$ of the first zero pivot $A_{ii}$ found. If no zero pivot is found it returns `rocsparse_status_success`.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_analysis_policy`
- `rocsparse_analysis_policy_reuse`
- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_create_mat_info`
- `rocsparse_csrsv_zero_pivot`
- `rocsparse_dcsrsv_analysis`
- `rocsparse_dcsrsv_buffer_size`
- `rocsparse_dcsrsv_solve`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_destroy_mat_info`
- `rocsparse_diag_type_non_unit`
- `rocsparse_fill_mode_lower`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_mat_descr`
- `rocsparse_mat_info`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_mat_diag_type`
- `rocsparse_set_mat_fill_mode`
- `rocsparse_set_pointer_mode`
- `rocsparse_solve_policy`
- `rocsparse_solve_policy_auto`
- `rocsparse_status`
- `rocsparse_status_zero_pivot`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
