# rocSPARSE Level-3 Matrix-Matrix Multiplication

## Description

This example illustrates the use of the `rocSPARSE` level 3 sparse matrix-dense matrix multiplication with a chosen sparse format (see: `rocsparse_spmm()` in [Key APIs and Concepts/rocSPARSE](#rocsparse)).

The operation calculates the following product:

$\hat{C} = \alpha \cdot op_a(A) \cdot op_b(B) + \beta \cdot C$

where

- $\alpha$ and $\beta$ are scalars
- $A$ is a sparse matrix
- $B$ and $C$ are dense matrices
- and $op_a(A)$ and $op_b(B)$ are the result of applying to matrices $A$ and $B$, respectively, one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

## Application flow

1. Set up a sparse matrix. Allocate an $A$ and a $B$ dense matrices and set up $\alpha$ and $\beta$ scalars.
2. Allocate device memory and copy input matrices from host to device.
3. Set up a handle.
4. Set up a matrix descriptors.
5. Prepare device for rocSPARSE SpMM invocation.
6. Perform analysis step.
7. Compute a sparse matrix multiplication.
8. Copy the result matrix from device to host.
9. Clear rocSPARSE allocations on device.
10. Clear device arrays.
11. Print result to the standard output.

## Key APIs and Concepts

### rocSPARSE

- `rocsparse_spmm(...)` performs a stage of sparse matrix-dense matrix multiplication. The current SpMM stage is defined by `rocsparse_spmm_stage`.

The following sparse matrix formats are supported: Blocked ELL, COO and CSR.

- `rocsparse_spmm_stage`: list of possible stages during SpMM computation. Typical order is `rocsparse_spmm_buffer_size`, `rocsparse_spmm_preprocess`, `rocsparse_spmm_compute`.
  - `rocsparse_spmm_stage_buffer_size` returns the required buffer size.
  - `rocsparse_spmm_stage_preprocess` preprocesses data.
  - `rocsparse_spmm_stage_compute` performs the actual SpMM computation.
  - `rocsparse_spmm_stage_auto`: automatic stage detection.
    - If `temp_buffer` is equal to `nullptr`, the required buffer size will be returned.
    - Otherwise, the SpMM preprocess and the SpMM algorithm will be executed.

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_pointer_mode` controls whether scalar parameters must be allocated on the host (`rocsparse_pointer_mode_host`) or on the device (`rocsparse_pointer_mode_device`). It is controlled by `rocsparse_set_pointer_mode`.
- `rocsparse_operation`: matrix operation applied to the given input matrix. The following values are accepted:
  - `rocsparse_operation_none`: identity operation $op(M) = M$.
  - `rocsparse_operation_transpose`: transpose operation $op(M) = M^\mathrm{T}$.
  - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $op(M) = M^\mathrm{H}$. This operation is not yet supported.

- `rocsparse_datatype`: data type of rocSPARSE matrix elements.
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

- `rocsparse_spmm_alg`: list of SpMM algorithms.
  - `rocsparse_spmm_alg_default`: default SpMM algorithm for the given format. For default algorithm, analysis step is required.
  - `rocsparse_spmm_alg_bell`: algorithm for Blocked ELL matrices.
  - `rocsparse_spmm_alg_coo_atomic`: atomic algorithm for COO matrices.
  - `rocsparse_spmm_alg_coo_segmented`: algorithm for COO matrices using segmented scan.
  - `rocsparse_spmm_alg_coo_segmented_atomic`: algorithm for COO format using segmented scan and atomics.
  - `rocsparse_spmm_alg_csr`: algorithm for CSR format using row split and shared memory.
  - `rocsparse_spmm_alg_csr_row_split`: algorithm for CSR format using row split and shfl
  - `rocsparse_spmm_alg_csr_merge`: algorithm for CSR format using conversion to COO.

- `rocsparse_spmat_descr`: sparse matrix descriptor.
- `rocsparse_create_[bell|coo|coo_aos|csr|csc|ell]_descr` creates a sparse matrix descriptor in BELL, COO or CSR format.

    We used COO format in the example.

    The descriptor should be destroyed at the end by `rocsparse_destroy_spmat_descr`.
- `rocsparse_destroy_spmat_descr`: Destroy a sparse matrix descriptor and release used resources allocated by the descriptor.

- `rocsparse_dnmat_descr` is a dense matrix descriptor.
- `rocsparse_create_dnmat_descr` creates a dense matrix descriptor.

    The descriptor should be destroyed at the end by `rocsparse_destroy_dnvec_descr`.
- `rocsparse_destroy_dnmat_descr` destroys a dense matrix descriptor.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_coo_descr`
- `rocsparse_create_dnmat_descr`
- `rocsparse_create_handle`
- `rocsparse_datatype`
- `rocsparse_datatype_f64_r`
- `rocsparse_destroy_dnmat_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_spmat_descr`
- `rocsparse_dnmat_descr`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
- `rocsparse_indextype`
- `rocsparse_indextype_i32`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_order_column`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`
- `rocsparse_spmat_descr`
- `rocsparse_spmm`
- `rocsparse_spmm_alg`
- `rocsparse_spmm_alg_default`
- `rocsparse_spmm_stage_buffer_size`
- `rocsparse_spmm_stage_compute`
- `rocsparse_spmm_stage_preprocess`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
