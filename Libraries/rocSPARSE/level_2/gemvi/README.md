# rocSPARSE Dense Matrix Sparse Vector Multiplication Example

## Description

This example showcases the usage of `rocsparse_gemvi` which multiplies a dense $m \times n$ matrix $A$ with a sparse vector $x$, scales this with $\alpha$, and adds the result to the dense the $\beta$-scaled vector $y$:

$y' = \alpha \cdot op(A) \cdot x + \beta \cdot y$

The example solves for $y'$ with the following concrete values:

$$
\overbrace{
  \left[
    \begin{matrix}
      245.7 \\
      245.7 \\
      370.3
    \end{matrix}
  \right]
}^{y'}
= \overbrace{3.7}^\alpha\cdot
\overbrace{
  \left[
    \begin{matrix}
       9.0 & 10.0 & 11.0 & 12.0 & 13.0 \\
      14.0 & 15.0 & 16.0 & 17.0 & 18.0 \\
      19.0 & 20.0 & 21.0 & 22.0 & 23.0 \\
    \end{matrix}
  \right]
}^A \cdot
\overbrace{
  \left[
    \begin{matrix}
      1.0 \\
      2.0 \\
      0.0 \\
      3.0 \\
      0.0
    \end{matrix}
  \right]
}^{x} + \overbrace{1.3}^\beta \cdot
\overbrace{
  \left[
    \begin{matrix}
      4.0 \\
      5.0 \\
      6.0
    \end{matrix}
  \right]
}^{y}
$$

## Application flow

1. Setup input:
   - Scalars $\alpha$ and $\beta$
   - Sparse vector $x$
   - Dense vector $y$
   - Dense matrix $A$
2. Prepare the device for the calculation by initializing handle.
3. Allocate and copy memory to device.
4. Execute the computation on device and copying the results to host.
5. Clean up the calculation environment by destroying the handle.
6. Free device memory.
7. Print the results.

## Key APIs and Concepts

### COO Storage Format

The sparse vector is stored in the coordinate (COO) storage format.
This works by storing the sparse vector $x$ as:

- the values $x_\text{values}$,
- the coordinates (indices) of said values into $x_\text{indices}$, and
- the amount of non zero values as $x_\text{non\_zero}$.

### rocSPARSE

- `rocsparse_[dscz]gemvi()`is the solver with four different function signatures depending on the type of the input matrix and vectors:
  - `d` double-precision real (`double`)
  - `s` single-precision real (`float`)
  - `c` single-precision complex (`rocsparse_float_complex`)
  - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_operation trans`: matrix operation type with the following options:
  - `rocsparse_operation_none`: identity operation: $op(M) = M$
  - `rocsparse_operation_transpose`: transpose operation: $op(M) = M^\mathrm{T}$
  - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $op(M) = M^\mathrm{H}$

  Currently, only `rocsparse_operation_none` is supported.

- `rocsparse_index_base idx_base`: base of indices
  - `rocsparse_index_base_zero`: zero based indexing
  - `rocsparse_index_base_one`: one based indexing

## Used API surface

### rocSPARSE

- `rocsparse_create_handle`
- `rocsparse_destroy_handle`
- `rocsparse_dgemvi_buffer_size`
- `rocsparse_dgemvi`
- `rocsparse_index_base::rocsparse_index_base_zero`
- `rocsparse_operation::rocsparse_operation_none`
- `rocsparse_pointer_mode_host`
- `rocsparse_set_pointer_mode`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
