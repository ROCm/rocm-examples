# rocSPARSE Preconditioner Tridiagonal Solver

## Description

This example illustrates the use of the `rocSPARSE` tridiagonal solver with multiple right hand sides:

$$ A \cdot X = B $$

where

- $A$ is a $m \times n$ tridiagonal matrix:

$$\begin{pmatrix}
  d_0    & u_0    & 0       & 0      & 0       & \cdots  & 0       \\
  l_1    & d_1    & u_1     & 0      & 0       & \cdots  & 0       \\
  0      & l_2    & d_2     & u_2    & 0       & \cdots  & 0       \\
  \vdots & \ddots & \ddots  & \ddots & \ddots  & \ddots  & \vdots  \\
  0      & \cdots & 0      & l_{m-2} & d_{m-2} & u_{m-2} & 0       \\
  0      & \cdots & 0      & 0       & l_{m-1} & d_{m-1} & u_{m-1} \\
  0      & \cdots & 0      & 0       & 0       & l_m     & d_m
\end{pmatrix}$$

- $B$ is dense right hand side matrix.
- $X$ is the dense solution matrix.

### Application flow

1. Set up input data.
2. Allocate device memory and offload input data to the device.
3. Initialize rocSPARSE by creating a handle.
4. Obtain the required buffer size.
5. Call dgtsv tridiagonal solver.
6. Copy result matrix to host.
7. Free rocSPARSE resources and device memory.
8. Check convergence.
9. Print result matrix, check errors.

## Key APIs and Concepts

The components of the tridiagonal system are stored in length $m$ vectors:

- upper diagonal with last elements 0: $\mathbf{u} = (u_0, u_1, \dots, u_{m-1}, 0)$
- main diagonal: $\mathbf{d} = (d_0, d_1, \dots, d_{m-1}, d_m)$
- lower diagonal with first elements 0: $\mathbf{l} = (0, l_1, \dots, l_{m-1}, l_m)$

### rocSPARSE

- rocSPARSE is initialized by calling `rocsparse_create_handle(rocsparse_handle*)` and is terminated by calling `rocsparse_destroy_handle(rocsparse_handle)`.
- `rocsparse_[sdcz]gtsv` computes the solution for a tridiagonal matrix with multiple right hand side. The correct function signature should be chosen based on the datatype of the input matrix:
  - `s` single-precision real (`float`)
  - `d` double-precision real (`double`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)
- `rocsparse_[sdcz]gtsv_buffer_size` allows to obtain the size (in bytes) of the temporary storage buffer required for the calculation. The character matched in `[sdcz]` coincides with the one matched in any of the mentioned functions.

## Demonstrated API Calls

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_destroy_handle`
- `rocsparse_dgtsv`
- `rocsparse_dgtsv_buffer_size`
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
