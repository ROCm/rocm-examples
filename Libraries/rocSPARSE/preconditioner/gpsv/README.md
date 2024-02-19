# rocSPARSE Preconditioner Pentadiagonal Solver

## Description

  This example illustrates the use of the `rocSPARSE` pentadiagonal solver with multiple batches:

  $$ A_i \cdot x_i = b_i $$

  where

- $s$ is the batch ID.
- $A_i$ is a $m \times n$ pentadiagonal matrix for each batch:

  $$\begin{pmatrix}
    d_0    & u_0    & w_0     & 0      & 0       & 0       & 0        & \cdots  & 0       \\
    l_1    & d_1    & u_1     & w_1    & 0       & 0       & 0        & \cdots  & 0       \\
    s_2    & l_2    & d_2     & u_2    & w_2     & 0       & 0        & \cdots  & 0       \\
    0      & s_3    & l_3     & d_3    & u_3     & w_3     & 0        & \cdots  & 0       \\
    \vdots & \ddots & \ddots  & \ddots & \ddots  & \ddots  & \ddots   & \cdots  & \vdots  \\
    0      & \cdots & 0       & s_{m-2} & l_{m-2} & d_{m-2} & u_{m-2} & w_{m-2} & 0       \\
    0      & \cdots & 0       & 0       & s_{m-1} & l_{m-1} & d_{m-1} & u_{m-1} & w_{m-1} \\
    0      & \cdots & 0       & 0       & 0       & s_m     & l_m     & d_m     & u_m
  \end{pmatrix}$$

  - $b_i$ is a dense right hand side vector for each batch.
  - $x_i$ is the dense solution vector for each batch.

### Application flow

  1. Set up input data.
  2. Allocate device memory and offload input data to the device.
  3. Initialize rocSPARSE by creating a handle.
  4. Obtain the required buffer size.
  5. Call interleaved batched pentadiagonal solver.
  6. Copy result matrix to host.
  7. Free rocSPARSE resources and device memory.
  8. Check convergence.
  9. Print result matrix, check errors.

## Key APIs and Concepts

  The components of the pentadiagonal system are stored in length $m$ vectors for each batch:

- second upper diagonal with last two elements 0: $\mathbf{w} = (w_0, w_1, \dots, w_{m-2}, 0, 0)$
- upper diagonal with last elements 0: $\mathbf{u} = (u_0, u_1, \dots, u_{m-1}, 0)$
- main diagonal: $\mathbf{d} = (d_0, d_1, \dots, d_{m-1}, d_m)$
- lower diagonal with first elements 0: $\mathbf{l} = (0, l_1, \dots, l_{m-1}, l_m)$
- lower diagonal with first two elements 0: $\mathbf{s} = (0, 0, s_2, \dots, s_{m-1}, s_m)$

### rocSPARSE

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_[sdcz]gpsv_interleaved_batch` computes the solution for a pentadiagonal matrix with multiple batches. The correct function signature should be chosen based on the datatype of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)
- `rocsparse_[sdcz]gpsv_interleaved_batch_buffer_size` allows to obtain the size (in bytes) of the temporary storage buffer required for the calculation. The character matched in `[sdcz]` coincides with the one matched in any of the mentioned functions.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_destroy_handle`
- `rocsparse_dgpsv_interleaved_batch`
- `rocsparse_dgpsv_interleaved_batch_buffer_size`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
