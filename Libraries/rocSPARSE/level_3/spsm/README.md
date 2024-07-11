# rocSPARSE Level 3 Triangular Solver Example

## Description

This example illustrates the use of the `rocSPARSE` level 3 triangular solver with a chosen sparse format.

The operation solves the following equation for $C$:

$op_a(A) \cdot C = \alpha \cdot op_b(B)$

where

- given a matrix $M$, $op_m(M)$ denotes one of the following:
  - $op_m(M) = M$ (identity)
  - $op_m(M) = M^T$ (transpose $M$: $M_{ij}^T = M_{ji}$)
  - $op_m(M) = M^H$ (conjugate transpose/Hermitian $M$: $M_{ij}^H = \bar M_{ji}$),
- $A$ is a sparse triangular matrix of order $m$ in CSR or COO format,
- $C$ is a dense matrix of size $m\times n$ containing the unknowns of the system,
- $B$ is a dense matrix of size $m\times n$ containing the right hand side of the equation,
- $\alpha$ is a scalar

## Application flow

1. Set up a sparse matrix in CSR format. Allocate matrices $B$ and $C$ and set up the scalar $\alpha$.
2. Prepare device for calculation.
3. Allocate device memory and copy input matrices from host to device.
4. Create matrix descriptors.
5. Do the calculation.
6. Copy the result matrix from device to host.
7. Clear rocSPARSE allocations on device.
8. Clear device memory.
9. Print result to the standard output.

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

- `rocsparse_spsm_alg`: list of SpSM algorithms.
  - `rocsparse_spsm_alg_default`: default SpSM algorithm for the given format (the only available option)

- `rocsparse_operation`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = M^\mathrm{H}$. This is currently not supported.

- `rocsparse_datatype`: data type of rocSPARSE vector and matrix elements.
  - `rocsparse_datatype_f32_r`: real 32-bit floating point type
  - `rocsparse_datatype_f64_r`: real 64-bit floating point type
  - `rocsparse_datatype_f32_c`: complex 32-bit floating point type
  - `rocsparse_datatype_f64_c`: complex 64-bit floating point type
  - `rocsparse_datatype_i8_r`: real 8-bit signed integer
  - `rocsparse_datatype_u8_r`: real 8-bit unsigned integer
  - `rocsparse_datatype_i32_r`: real 32-bit signed integer
  - `rocsparse_datatype_u32_r` real 32-bit unsigned integer

- `rocsparse_indextype` indicates the index type of a rocSPARSE index vector.
  - `rocsparse_indextype_u16`: 16-bit unsigned integer
  - `rocsparse_indextype_i32`: 32-bit signed integer
  - `rocsparse_indextype_i64`: 64-bit signed integer

- `rocsparse_index_base` indicates the index base of indices.
  - `rocsparse_index_base_zero`: zero based indexing.
  - `rocsparse_index_base_one`: one based indexing.

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
- `rocsparse_spmat_descr`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
- `rocsparse_indextype`
- `rocsparse_indextype_i32`
- `rocsparse_int`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_order`
- `rocsparse_order_column`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`
- `rocsparse_spmat_descr`
- `rocsparse_spsm`
- `rocsparse_spsm_alg`
- `rocsparse_spsm_alg_default`
- `rocsparse_spsm_stage`
- `rocsparse_spsm_stage_buffer_size`
- `rocsparse_spsm_stage_compute`
- `rocsparse_spsm_stage_preprocess`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipMemset`
