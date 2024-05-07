# rocSPARSE Level 2 BSR Triangular Solver Example

## Description
This example illustrates the use of the `rocSPARSE` level 2 triangular solver using the BSR storage format.

This triangular solver is used to solve a linear system of the form

$$
A'y = \alpha x,
$$

where

- $A$ is a sparse triangular matrix of order $n$ whose elements are the coefficients of the equations,
- $A'$ is one of the following:
    - $A' = A$ (identity)
    - $A' = A^T$ (transpose $A$: $A_{ij}^T = A_{ji}$)
    - $A' = A^H$ (conjugate transpose/Hermitian $A$: $A_{ij}^H = \bar A_{ji}$),
- $\alpha$ is a scalar,
- $x$ is a dense vector of size $m$ containing the constant terms of the equations, and
- $y$ is a dense vector of size $n$ which contains the unknowns of the system.

Obtaining the solution for such a system consists of finding concrete values of all the unknowns such that the above equality holds.

### Application flow
1. Setup input data.
2. Allocate device memory and offload input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare utility variables for rocSPARSE bsrsv invocation.
5. Perform analysis step.
6. Perform triangular solve $Ay = \alpha x$.
7. Check results obtained. If no zero-pivots, copy solution vector $y$ from device to host and compare with expected result.
8. Free rocSPARSE resources and device memory.
9. Print validation result.

## Key APIs and Concepts
### BSR Matrix Storage Format
The [Block Compressed Sparse Row (BSR) storage format](https://rocsparse.readthedocs.io/en/latest/usermanual.html#bsr-storage-format) describes a sparse matrix using three arrays. The idea behind this storage format is to split the given sparse matrix into equal sized blocks of dimension `bsr_dim` and store those using the [CSR format](https://rocsparse.readthedocs.io/en/latest/usermanual.html#csr-storage-format). Because the CSR format only stores non-zero elements, the BSR format introduces the concept of __non-zero block__: a block that contains at least one non-zero element. Note that all elements of non-zero blocks are stored, even if some of them are equal to zero.

Therefore, defining
- `mb`: number of rows of blocks
- `nb`: number of columns of blocks
- `nnzb`: number of non-zero blocks
- `bsr_dim`: dimension of each block

we can describe a sparse matrix using the following arrays:
- `bsr_val`: contains the elements of the non-zero blocks of the sparse matrix. The elements are stored block by block in column- or row-major order. That is, it is an array of size `nnzb` $\cdot$ `bsr_dim` $\cdot$ `bsr_dim`.

- `bsr_row_ptr`: given $i \in [0, mb]$
    - if $` 0 \leq i < mb `$, `bsr_row_ptr[i]` stores the index of the first non-zero block in row $i$ of the block matrix
    - if $i = mb$, `bsr_row_ptr[i]` stores `nnzb`.

    This way, row $j \in [0, mb)$ contains the non-zero blocks of indices from `bsr_row_ptr[j]` to `bsr_row_ptr[j+1]-1`. The corresponding values in `bsr_val` can be accessed from `bsr_row_ptr[j] * bsr_dim * bsr_dim` to `(bsr_row_ptr[j+1]-1) * bsr_dim * bsr_dim`.

- `bsr_col_ind`: given $i \in [0, nnzb-1]$, `bsr_col_ind[i]` stores the column of the $i^{th}$ non-zero block in the block matrix.

Note that, for a given $m\times n$ matrix, if the dimensions are not evenly divisible by the block dimension then zeros are padded to the matrix so that $mb$ and $nb$ are the smallest integers greater than or equal to $`\frac{m}{\texttt{bsr\_dim}}`$ and $`\frac{n}{\texttt{bsr\_dim}}`$, respectively.

For instance, consider a sparse matrix as

$$
A=
\left(
\begin{array}{cccc:cccc:cc}
8 & 0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
7 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
2 & 5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 5 \\
7 & 7 & 7 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 9 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 5 & 6 & 4 & 0 & 0 & 0 & 0 \\
\end{array}
\right)
$$

Taking $`\texttt{bsr\_dim} = 4`$, we can represent $A$ as an $mb \times nb$ block matrix

$$
A=
\begin{pmatrix}
A_{00} & O & O \\
A_{10} & O & A_{12} \\
A_{20} & A_{21} & O
\end{pmatrix}
$$

with the following non-zero blocks:

$$
\begin{matrix}
A_{00}=
\begin{pmatrix}
8 & 0 & 2 & 0\\
0 & 3 & 0 & 0\\
7 & 0 & 1 & 0\\
2 & 5 & 0 & 0
\end{pmatrix} &
A_{10}=
\begin{pmatrix}
4 & 0 & 0 & 0 \\
7 & 7 & 7 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} \\
\\
A_{12}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
5 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} &
A_{20}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 5 \\
0 & 0 & 0 & 0
\end{pmatrix} \\
\\
A_{21}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
9 & 0 & 0 & 0 \\
6 & 4 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} &
\end{matrix}
$$

and the zero matrix:

$$
O = 0_4=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

Therefore, the BSR representation of $A$, using column-major ordering, is:

```
bsr_val = { 8, 0, 7, 2, 0, 3, 0, 5, 2, 0, 1, 0, 0, 0, 0, 0   // A_{00}
            4, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0   // A_{10}
            0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // A_{12}
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0   // A_{20}
            0, 9, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 } // A_{21}

bsr_row_ptr = { 0, 1, 3, 4 }

bsr_col_ind = { 0, 0, 2, 0, 1 }
```

### rocSPARSE
- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_pointer_mode` controls whether scalar parameters must be allocated on the host (`rocsparse_pointer_mode_host`) or on the device (`rocsparse_pointer_mode_device`). It is controlled by `rocsparse_set_pointer_mode`.
- `rocsparse_direction dir`: matrix storage of BSR blocks. The following values are accepted:
    - `rocsparse_direction_row`: parse blocks by rows.
    - `rocsparse_direction_column`: parse blocks by columns.
- `rocsparse_operation trans`: matrix operation applied to the given input matrix. The following values are accepted:
    - `rocsparse_operation_none`: identity operation $A' = A$.
    - `rocsparse_operation_transpose`: transpose operation $A' = A^\mathrm{T}$.
    - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $A' = A^\mathrm{H}$. This operation is not yet supported.
- `rocsparse_mat_descr descr`: holds all properties of a matrix. The properties set in this example are the following:
    - `rocsparse_diag_type`: indicates whether the diagonal entries of a matrix are unit elements (`rocsparse_diag_type_unit`) or not (`rocsparse_diag_type_non_unit`).
    - `rocsparse_fill_mode`: indicates whether a (triangular) matrix is lower (`rocsparse_fill_mode_lower`) or upper (`rocsparse_fill_mode_upper`) triangular.
- `rocsparse_[sdcz]bsrsv_buffer_size` allows to obtain the size (in bytes) of the temporary storage buffer required for the `rocsparse_[sdcz]bsrsv_analysis` and `rocsparse_[sdcz]bsrsv_solve` functions. The character matched in `[sdcz]` coincides with the one matched in any of the mentioned functions.
- `rocsparse_solve_policy policy`: specifies the policy to follow for triangular solvers and factorizations. The only value accepted is `rocsparse_solve_policy_auto`.
- `rocsparse_[sdcz]bsrsv_solve` solves a sparse triangular linear system $A'y = \alpha x$. The correct function signature should be chosen based on the datatype of the input matrix:
    - `s` single-precision real (`float`)
    - `d` double-precision real (`double`)
    - `c` single-precision complex (`rocsparse_float_complex`)
    - `z` double-precision complex (`rocsparse_double_complex`)
- `rocsparse_analysis_policy analysis`: specifies the policy to follow for analysis data. The following values are accepted:
    - `rocsparse_analysis_policy_reuse`: the analysis data gathered is re-used.
    - `rocsparse_analysis_policy_force`: the analysis data will be re-built.
- `rocsparse_[sdcz]bsrsv_analysis` performs the analysis step for `rocsparse_[sdcz]bsrsv_solve`. The character matched in `[sdcz]` coincides with the one matched in `rocsparse_[sdcz]bsrsv_solve`.
- `rocsparse_bsrsv_zero_pivot(rocsparse_handle, rocsparse_mat_info, rocsparse_int *position)` returns `rocsparse_status_zero_pivot` if either a structural or numerical zero has been found during the execution of `rocsparse_[sdcz]bsrsv_solve(....)` and stores in `position` the index $i$ of the first zero pivot $A_{ii}$ found. If no zero pivot is found it returns `rocsparse_status_success`.

## Demonstrated API Calls

### rocSPARSE
- `rocsparse_analysis_policy`
- `rocsparse_analysis_policy_reuse`
- `rocsparse_bsrsv_zero_pivot`
- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_create_mat_info`
- `rocsparse_dbsrsv_analysis`
- `rocsparse_dbsrsv_buffer_size`
- `rocsparse_dbsrsv_solve`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_destroy_mat_info`
- `rocsparse_diag_type_non_unit`
- `rocsparse_direction`
- `rocsparse_direction_column`
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
