# rocSPARSE Level 2 Triangular Solver Example
## Description
This example illustrates the use of the `rocSPARSE` level 2 triangular solver with a chosen sparse format.

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
- $x$ is a dense vector of size $n$ containing the constant terms of the equations, and
- $y$ is a dense vector of size $n$ which contains the unknowns of the system.

### Application flow
1. Setup input data.
2. Allocate device memory and offload input data to device.
3. Initialize rocSPARSE by creating a handle.
4. Prepare device for rocSPARSE SpSV invocation.
5. Perform analysis step.
6. Perform triangular solve $Ay = \alpha x$.
7. Copy solution vector $y$ from device to host.
8. Free rocSPARSE resources and device memory.
9. Print solution vector $y$ to the standard output.

## Key APIs and Concepts
### rocSPARSE
- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_pointer_mode` controls whether scalar parameters must be allocated on the host (`rocsparse_pointer_mode_host`) or on the device (`rocsparse_pointer_mode_device`). It is controlled by `rocsparse_set_pointer_mode`.
- `rocsparse_operation`: matrix operation applied to the given input matrix. The following values are accepted:
    - `rocsparse_operation_none`: identity operation $A' = A$.
    - `rocsparse_operation_transpose`: transpose operation $A' = A^\mathrm{T}$.
    - `rocsparse_operation_conjugate_transpose`: conjugate transpose operation (Hermitian matrix) $A' = A^\mathrm{H}$. This operation is not yet supported.

- `rocsparse_spsv()` solves a sparse triangular linear system of a sparse matrix in CSR or COO format.

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

- `rocsparse_spsv_alg`: list of SpSV algorithms.
    - `rocsparse_spsv_alg_default`: default SpSV algorithm for the given format (the only available option)

- `rocsparse_spmat_descr`: sparse matrix descriptor.
- `rocsparse_create_[coo|csr]_descr` creates a sparse matrix descriptor in COO or CSR format. 

    We used COO format in the example.

    The descriptor should be destroyed at the end by `rocsparse_destroy_spmat_descr`.
- `rocsparse_destroy_spmat_descr`: Destroy a sparse matrix descriptor and release used resources allocated by the descriptor.

- `rocsparse_dnvec_descr` is a dense vector descriptor.
- `rocsparse_create_dnvec_descr` creates a dense vector descriptor. 

    The descriptor should be destroyed at the end by `rocsparse_destroy_dnvec_descr`.
- `rocsparse_destroy_dnvec_descr` destroys a dense vector descriptor.

-  `rocsparse_spsv_stage`: list of possible stages during SpSV computation. Typical order is `rocsparse_spsv_buffer_size`, `rocsparse_spsv_preprocess`, `rocsparse_spsv_compute`.
    - `rocsparse_spsv_stage_buffer_size` returns the required buffer size.
    - `rocsparse_spsv_stage_preprocess` preprocesses data.
    - `rocsparse_spsv_stage_compute` performs the actual SpSV computation.
    - `rocsparse_spsv_stage_auto`: automatic stage detection.
        - If `temp_buffer` is equal to `nullptr`, the required buffer size will be returned. 
        - If `buffer_size` is equal to `nullptr`, analysis will be performed.
        - Otherwise, the SpSV preprocess and the SpSV algorithm will be executed.

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
- `rocsparse_spsv`
- `rocsparse_spsv_alg`
- `rocsparse_spsv_alg_default`
- `rocsparse_spsv_stage_buffer_size`
- `rocsparse_spsv_stage_compute`
- `rocsparse_spsv_stage_preprocess`

### HIP runtime
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
