# rocSPARSE Level 2 Matrix-Vector Multiplication Example

## Description

This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication with a chosen sparse format (see: `rocsparse_spmv()` in [Key APIs and Concepts/rocSPARSE](#rocsparse)).

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot op(A) \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $A$ is a sparse matrix
- $op(A)$ is the result of applying to matrix $A$ one of the `rocsparse_operation` described below in [Key APIs and Concepts - rocSPARSE](#rocsparse).

### Application flow

1. Set up input data.
2. Allocate device memory and offload input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare device for rocSPARSE SpMV invocation.
5. Perform preprocessing step.
6. Compute a sparse matrix-vector multiplication.
7. Copy the result vector from device to host.
8. Free rocSPARSE resources and device memory.
9. Print result to the standard output.

## Key APIs and Concepts

### rocSPARSE

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_operation`: matrix operation applied to the given input matrix. The following values are accepted:
  - `rocsparse_operation_none`: identity operation $op(M) = M$.
  - `rocsparse_operation_transpose`: transpose operation $op(M) = M^\mathrm{T}$.
  - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $op(M) = M^\mathrm{H}$.

- `rocsparse_spmv()` solves a sparse matrix-vector product in the following formats: BELL, BSR, COO, COO AoS, CSR, CSC and ELL.

- `rocsparse_datatype`: data type of rocSPARSE vector and matrix elements.
  - `rocsparse_datatype_f32_r`: real 32-bit floating point type
  - `rocsparse_datatype_f64_r`: real 64-bit floating point type
  - `rocsparse_datatype_f32_c`: complex 32-bit floating point type
  - `rocsparse_datatype_f64_c`: complex 64-bit floating point type
  - `rocsparse_datatype_i32_r`: real 32-bit signed integer

  Mixed precision is available as:

  |  $A$ and $\mathbf{x}$     |       $\mathbf{y}$         |       `compute_type`       |
  |---------------------------|----------------------------|----------------------------|
  | `rocsparse_datatype_i8_r` | `rocsparse_datatype_i32_r` | `rocsparse_datatype_i32_r` |
  | `rocsparse_datatype_i8_r` | `rocsparse_datatype_f32_r` | `rocsparse_datatype_f32_r` |
  | `rocsparse_datatype_i8_r` | `rocsparse_datatype_i32_r` | `rocsparse_datatype_i32_r` |

  |             $A$            | $\mathbf{x}$ , $\mathbf{y}$ and `compute_type` |
  |----------------------------|------------------------------------------------|
  | `rocsparse_datatype_f32_r` |            `rocsparse_datatype_i32_r`          |
  | `rocsparse_datatype_f64_r` |            `rocsparse_datatype_i64_r`          |

- `rocsparse_indextype` indicates the index type of a rocSPARSE index vector.
  - `rocsparse_indextype_i32`: 32-bit signed integer
  - `rocsparse_indextype_i64`: 64-bit signed integer

- `rocsparse_index_base` indicates the index base of indices.
  - `rocsparse_index_base_zero`: zero based indexing.
  - `rocsparse_index_base_one`: one based indexing.

- `rocsparse_spmv_alg`: list of SpMV algorithms.
  - `rocsparse_spmv_alg_default`: default SpMV algorithm for the given format. For default algorithm, analysis step is required.
  - `rocsparse_spmv_alg_bell`: algorithm for BELL matrices
  - `rocsparse_spmv_alg_bsr`: algorithm for BSR matrices
  - `rocsparse_spmv_alg_coo`: segmented algorithm for COO matrices
  - `rocsparse_spmv_alg_coo_atomic`: atomic algorithm for COO matrices
  - `rocsparse_spmv_alg_csr_adaptive`: adaptive algorithm for CSR and CSC matrices
  - `rocsparse_spmv_alg_csr_stream`: stream algorithm for CSR and CSC matrices
  - `rocsparse_spmv_alg_ell`: algorithm for ELL matrices

  The default algorithm for CSR and CSC matrices is `rocsparse_spmv_alg_csr_adaptive`.

  The default algorithm for COO and COO AoS is `rocsparse_spmv_alg_coo` (segmented).

- `rocsparse_spmat_descr`: sparse matrix descriptor.
- `rocsparse_create_[bell|coo|coo_aos|csr|csc|ell]_descr` creates a sparse matrix descriptor in BELL, COO, COO AoS, CSR, CSC or ELL format.

  We used COO format in the example.

  The descriptor should be destroyed at the end by `rocsparse_destroy_spmat_descr`.

- `rocsparse_destroy_spmat_descr`: Destroy a sparse matrix descriptor and release used resources allocated by the descriptor.

- `rocsparse_dnvec_descr` is a dense vector descriptor.
- `rocsparse_create_dnvec_descr` creates a dense vector descriptor.

    The descriptor should be destroyed at the end by `rocsparse_destroy_dnvec_descr`.

- `rocsparse_destroy_dnvec_descr` destroys a dense vector descriptor.

- `rocsparse_spmv_stage`: list of possible stages during SpMV computation. Typical order is `rocsparse_spmv_buffer_size`, `rocsparse_spmv_preprocess`, `rocsparse_spmv_compute`.
  - `rocsparse_spmv_stage_buffer_size` returns the required buffer size.
  - `rocsparse_spmv_stage_preprocess` preprocesses data.
  - `rocsparse_spmv_stage_compute` performs the actual SpMV computation.
  - `rocsparse_spmv_stage_auto`: automatic stage detection.
    - If `temp_buffer` is equal to `nullptr`, the required buffer size will be returned.
    - Otherwise, the SpMV preprocess and the SpMV algorithm will be executed.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_coo_descr`
- `rocsparse_create_dnvec_descr`
- `rocsparse_create_handle`
- `rocsparse_datatype`
- `rocsparse_datatype_f64_r`
- `rocsparse_destroy_dnvec_descr`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_spmat_descr`
- `rocsparse_dnvec_descr`
- `rocsparse_handle`
- `rocsparse_index_base`
- `rocsparse_index_base_zero`
- `rocsparse_indextype`
- `rocsparse_indextype_i32`
- `rocsparse_int`
- `rocsparse_operation`
- `rocsparse_operation_none`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`
- `rocsparse_spmat_descr`
- `rocsparse_spmv_alg`
- `rocsparse_spmv_alg_default`
- `rocsparse_spmv_ex`
- `rocsparse_spmv_stage_buffer_size`
- `rocsparse_spmv_stage_compute`
- `rocsparse_spmv_stage_preprocess`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
