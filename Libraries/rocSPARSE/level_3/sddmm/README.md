# rocSPARSE Level-3 Sampled Dense-Dense Matrix Multiplication Example

## Description

This example illustrates the use of the `rocSPARSE` sampled dense-dense matrix multiplication.

The operation solves the following equation for C:

<!-- markdownlint-disable-next-line no-space-in-emphasis -->
$\C := \alpha(\cdot op_a(A) * \cdot op_b(B)) * sppat(\cdot C) + \beta C $

where

- $\alpha$ and $\beta$ are scalars
- $A$ and $B$ are dense matrices
- $C$ is a sparse matrix in CSR format, the result will be in this matrix
- $sppat(C)$ is the sparsity pattern of C matrix
- and $op_a(A)$ and $op_b(B)$ are the result of applying to matrices $A$ and $B$, respectively, one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

## Application flow

1. Set up a sparse matrix in CSR format. Allocate an $A$ and a $B$ matrices and set up $\alpha$ and $\beta$ scalars.
2. Prepare device for calculation.
3. Allocate device memory and copy input matrices from host to device.
4. Create matrix descriptors.
5. Allocate temporary buffer.
6. Do the pre calculation.
7. Do the calculation.
8. Copy the result matrix from device to host.
9. Clear rocSPARSE allocations on device.
10. Clear device memory.
11. Print result to the standard output.

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

- `rocsparse_spsm(...)` performs a sparse matrix-dense matrix multiplication. This single function is used to run all three stages of the calculation.
  - `rocsparse_spsm_stage_buffer_size` will query the size of the temporary buffer, which will hold the data between the preprocess and the compute stages.
  - `rocsparse_spsm_stage_preprocess` will preprocess the data and save it in the temporary buffer.
  - `rocsparse_spsm_stage_compute` will do the actual spsm calculation.
  - `rocsparse_spsm_stage_auto` will figure out the current stage and run it. If `temp_buffer` pointer equals nullptr the stage will be `rocsparse_spsm_stage_buffer_size`, if `buffer_size` the function will perform the `rocsparse_spsm_stage_preprocess` stage, otherwise it will perform the `rocsparse_spsm_stage_compute`.
  - The `rocsparse_spsm_stage_buffer_size` and `rocsparse_spsm_stage_compute` stages are asynchronous. It is the callers responsibility to assure that these stages are complete, before using their result. `rocsparse_spsm_stage_preprocess` will run in a blocking manner, no synchronization is necessary.

- `rocsparse_operation`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = M^\mathrm{H}$. This is currently not supported.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_csr_descr`
- `rocsparse_create_dnmat_descr`
- `rocsparse_create_handle`
- `rocsparse_datatype`
- `rocsparse_datatype_f64_r`
- `rocsparse_destroy_dnmat_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_spmat_descr`
- `rocsparse_dnmat_descr`
- `rocsparse_handle`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
- `rocsparse_indextype`
- `rocsparse_indextype_i32`
- `rocsparse_int`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_order`
- `rocsparse_order_row`
- `rocsparse_pointer_mode_host`
- `rocsparse_sddmm`
- `rocsparse_sddmm_alg_default`
- `rocsparse_sddmm_buffer_size`
- `rocsparse_sddmm_preprocess`
- `rocsparse_set_pointer_mode`
- `rocsparse_spmat_descr`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
