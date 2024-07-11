# rocSPARSE Preconditioner CSR Iterative Incomplete LU Decomposition Example

## Description

This example illustrates the use of the `rocSPARSE` iterative incomplete LU factorization preconditioner using the CSR storage format.

Given an arbitrary matrix $A$ of order $n$, computing its LU decomposition consists of finding a lower triangular matrix $L$ and a upper triangular matrix $U$ such that
$$A = L \cdot U.$$

The _incomplete_ LU decomposition is a sparse approximation of the above-mentioned LU decomposition. `rocSPARSE` allows us to iteratively compute a sparse lower triangular matrix $L$ and a sparse upper triangular matrix $U$ such that
$$A \approx L \cdot U.$$

### Application flow

1. Set up input data.
2. Allocate device memory and offload input data to the device.
3. Initialize rocSPARSE and prepare utility variables for csritilu0 invocation.
4. Sort CSR matrix.
5. Query the required buffer size in bytes for the iterative-ILU0-related functions.
6. Perform the preprocessing.
7. Perform the iterative incomplete LU factorization.
8. Fetch the convergence data.
9. Check errors and print the resulting matrices.
10. Free rocSPARSE resources and device memory.
11. Print validation result.

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
- `rocsparse_mat_descr descr`: holds all properties of a matrix.
- `rocsparse_itilu0_alg` represents an iterative ILU0 (Zero Fill-in Incomplete LU) algorithm. The following values are accepted:
  - `rocsparse_itilu0_alg_async_inplace`: Asynchronous iterative ILU0 algorithm with in-place storage.
  - `rocsparse_itilu0_alg_async_split`: Asynchronous iterative ILU0 algorithm with explicit storage splitting.
  - `rocsparse_itilu0_alg_sync_split`: Synchronous iterative ILU0 algorithm with explicit storage splitting.
  - `rocsparse_itilu0_alg_sync_split_fusion`: Semi-synchronous iterative ILU0 algorithm with explicit storage splitting.
  - `rocsparse_itilu0_alg_default`: same as `rocsparse_itilu0_alg_async_inplace`.
- `rocsparse_itilu0_option`: available options to perform the iterative ILU0 algorithm. The following values are accepted:
  - `rocsparse_itilu0_option_verbose`
  - `rocsparse_itilu0_option_stopping_criteria`: Compute a stopping criteria.
  - `rocsparse_itilu0_option_compute_nrm_correction`: Compute correction.
  - `rocsparse_itilu0_option_compute_nrm_residual`: Compute residual.
  - `rocsparse_itilu0_option_convergence_history`: Log convergence history.
  - `rocsparse_itilu0_option_coo_format`: Use internal coordinate format.
- `rocsparse_index_base idx_base` indicates the index base of the indices. The following values are accepted:
  - `rocsparse_index_base_zero`: zero based indexing.
  - `rocsparse_index_base_one`: one based indexing.
- `rocsparse_datatype`: data type of rocSPARSE vector and matrix elements.
  - `rocsparse_datatype_f32_r`: real 32-bit floating point type
  - `rocsparse_datatype_f64_r`: real 64-bit floating point type
  - `rocsparse_datatype_f32_c`: complex 32-bit floating point type
  - `rocsparse_datatype_f64_c`: complex 64-bit floating point type
  - `rocsparse_datatype_i8_r`: real 8-bit signed integer
  - `rocsparse_datatype_u8_r`: real 8-bit unsigned integer
  - `rocsparse_datatype_i32_r`: real 32-bit signed integer
  - `rocsparse_datatype_u32_r` real 32-bit unsigned integer

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

- `rocsparse_[sdcz]csritilu0_compute` computes iteratively the incomplete LU factorization of a sparse CSR matrix $A$, such that $A \approx L \cdot U$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

    The matrix $A$ must be sorted beforehand.

- `rocsparse_csritilu0_buffer_size` computes the size in bytes of the buffer needed by `rocsparse_csritilu0_preprocess`, `rocsparse_[sdcz]csritilu0_compute` and `rocsparse_csritilu0_history`. The matrix $A$ must be sorted beforehand.

- `rocsparse_csritilu0_preprocess` computes the information required to run `rocsparse_[sdcz]csritilu0_compute` and stores it in the buffer. The matrix $A$ must be sorted beforehand.

- `rocsparse_[sdcz]csritilu0_history` fetches the convergence history data. The character matched in `[sdcz]` coincides with the one matched in `rocsparse_[sdcz]csritilu0_compute`. The matrix $A$ must be sorted beforehand.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_create_identity_permutation`
- `rocsparse_create_mat_descr`
- `rocsparse_csritilu0_buffer_size`
- `rocsparse_csritilu0_preprocess`
- `rocsparse_csrsort`
- `rocsparse_csrsort_buffer_size`
- `rocsparse_datatype`
- `rocsparse_datatype_f64_r`
- `rocsparse_dcsritilu0_compute`
- `rocsparse_dcsritilu0_history`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_dgthr`
- `rocsparse_handle`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
- `rocsparse_int`
- `rocsparse_itilu0_alg`
- `rocsparse_itilu0_alg_default`
- `rocsparse_mat_descr`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
