# rocSPARSE Level 3 CSR Triangular Solver Example

## Description

This example illustrates the use of the `rocSPARSE` level 3 triangular solver using the CSR storage format.

This triangular solver is used to solve a linear system of the form

$$
op_a(A) \cdot op_b(X) = \alpha \cdot op_b(B).
$$

where

- $A$ is a sparse triangular matrix of order $n$ whose elements are the coefficients of the equations,
- given a matrix $M$, $op_m(M)$ denotes one of the following:
  - $op_m(M) = M$ (identity)
  - $op_m(M) = M^T$ (transpose $M$: $M_{ij}^T = M_{ji}$)
  - $op_m(M) = M^H$ (conjugate transpose/Hermitian $M$: $M_{ij}^H = \bar M_{ji}$),
- $X$ is a dense matrix of size $n \times nrhs$ containing the unknowns of the systems,
- $\alpha$ is a scalar,
- $B$ is a dense matrix of size $n \times nrhs$ containing the right hand sides of the equations,
- the operation performed on $B$ and $X$, $op_b$, must be the same.

Obtaining the solution for such a system consists of finding concrete values of all the unknowns such that the above equalities holds.

This is the same as solving the classical system of linear equations $op_a(A) x_i = \alpha b_i$ for each $i\in[0, nrhs-1]$, where $x_i$ and $b_i$ are the $i$-th rows or columns of $X$ and $B$, depending on the operation performed on $X$ and $B$. This is showcased in [level 2 example csrsv](../../level_2/csrsv/README.md).

### Application flow

1. Set up input data.
2. Allocate device memory and copy input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare utility variables for rocSPARSE csrsm invocation.
5. Perform analysis step.
6. Sort CSR matrix before calling the solver function.
7. Call to rocSPARSE csrsm to solve the linear system.
8. Check results. If no zero-pivots, copy solution matrix $X$ from device to host and compare with expected result.
9. Free rocSPARSE resources and device memory.
10. Print validation result.

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

- `rocsparse_operation trans`: matrix operation applied to the given matrix. The following values are accepted:

  - `rocsparse_operation_none`: identity operation $op(M) = M$.
  - `rocsparse_operation_transpose`: transpose operation $op(M) = M^\mathrm{T}$.
  - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $op(M) = M^\mathrm{H}$.

  Currently, only `rocsparse_operation_none` and `rocsparse_operation_transpose` are supported for both $A$ and $B$ matrices.

- `rocsparse_mat_descr descr`: holds all properties of a matrix. The properties set in this example are the following:

  - `rocsparse_diag_type`: indicates whether the diagonal entries of a matrix are unit elements (`rocsparse_diag_type_unit`) or not (`rocsparse_diag_type_non_unit`).
  - `rocsparse_fill_mode`: indicates whether a (triangular) matrix is lower (`rocsparse_fill_mode_lower`) or upper (`rocsparse_fill_mode_upper`) triangular.

- `rocsparse_index_base idx_base` indicates the index base of the indices. The following values are accepted:
  - `rocsparse_index_base_zero`: zero based indexing.
  - `rocsparse_index_base_one`: one based indexing.

- A matrix stored using CSR format is _sorted_ if the order of the elements in the values array `csr_val` is such that the column indexes in `csr_col_ind` are in (strictly) increasing order for each row. Otherwise the matrix is _unsorted_.

- `rocsparse_csrsort` permutes and sorts a matrix in CSR format. A permutation $\sigma$ is applied to the CSR column indices array, such that the sorting is performed based the permuted array `csr_col_ind_perm` = $\sigma ($ `csr_col_ind` $)$. In this example, $\sigma$ is set as the identity permutation by calling `rocsparse_create_identity_permutation`.

- `rocsparse_[sdcz]gthr` gathers elements from a dense vector $y$ and stores them into a sparse vector $x$. In this example, we take $x = y =$ `csr_val` and $x[i] = y[\hat\sigma(i)]$ for $i \in [0, \texttt{nnz}-1]$, where $\hat\sigma$ is the composition of the sorting explained above with the permutation $\sigma$.

  The correct function signature should be chosen based on the datatype of the input vector:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_csrsort_buffer_size` provides the size of the temporary storage buffer required by `rocsparse_csrsort`.

- `rocsparse_create_identity_permutation` initializes a given permutation vector of size $n$ as the identity permutation $`\begin{pmatrix} 0 & 1 & 2 & \cdots & n-1 \end{pmatrix}`$.

- `rocsparse_solve_policy policy`: specifies the policy to follow for triangular solvers and factorizations. The only value accepted is `rocsparse_solve_policy_auto`.

- `rocsparse_[sdcz]csrsm_solve` solves a sparse triangular linear system $A X = \alpha B$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

  The matrix $A$ must be sorted beforehand.

- `rocsparse_[sdcz]csrsm_buffer_size` allows to obtain the size (in bytes) of the temporary storage buffer required for the `rocsparse_[sdcz]csrsm_analysis` and `rocsparse_[sdcz]csrsm_solve` functions. The character matched in `[sdcz]` coincides with the one matched in any of the mentioned functions.

- `rocsparse_analysis_policy analysis`: specifies the policy to follow for analysis data. The following values are accepted:
  - `rocsparse_analysis_policy_reuse`: the analysis data gathered is re-used.
  - `rocsparse_analysis_policy_force`: the analysis data will be re-built.

- `rocsparse_[sdcz]csrsm_analysis` performs the analysis step for `rocsparse_[sdcz]csrsm_solve`. The character matched in `[sdcz]` coincides with the one matched in `rocsparse_[sdcz]csrsm_solve`.

- `rocsparse_csrsm_zero_pivot(rocsparse_handle, rocsparse_mat_info, rocsparse_int *position)` returns `rocsparse_status_zero_pivot` if either a structural or numerical zero has been found during the execution of `rocsparse_[sdcz]csrsm_solve(....)` and stores in `position` the index $i$ of the first zero pivot $A_{ii}$ found. If no zero pivot is found it returns `rocsparse_status_success`.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_analysis_policy`
- `rocsparse_analysis_policy_reuse`
- `rocsparse_create_handle`
- `rocsparse_create_identity_permutation`
- `rocsparse_create_mat_descr`
- `rocsparse_create_mat_info`
- `rocsparse_csrsm_zero_pivot`
- `rocsparse_csrsort`
- `rocsparse_csrsort_buffer_size`
- `rocsparse_dcsrsm_analysis`
- `rocsparse_dcsrsm_buffer_size`
- `rocsparse_dcsrsm_solve`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_destroy_mat_info`
- `rocsparse_dgthr`
- `rocsparse_diag_type_non_unit`
- `rocsparse_fill_mode_lower`
- `rocsparse_handle`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
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

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
