# rocSPARSE Level 2 BSR Matrix--Vector Multiplication
## Description
This example illustrates the use of the `rocSPARSE` level 2 sparse matrix--vector multiplication using BSR storage format.

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot A' \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $A'$ is a sparse matrix in BSR format with `rocsparse_operation` and described below.

## Application flow
1. Setup a sparse matrix in BSR format. Allocate an x and a y vector and set up $\alpha$ and $\beta$ scalars.
2. Setup a handle, a matrix descriptor and a matrix info.
3. Allocate device memory and copy input matrix and vectors from host to device.
4. Compute a sparse matrix multiplication, using BSR storage format.
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts
### BSR Matrix Format

We are working with Block Compressed Row Format (BSR) matrices in the recent example.

Consider a sparse matrix as

$$
A=
\left[
\begin{array}{ccc:ccc:ccc:c}
8 & 0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
7 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
\hline
2 & 5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 5 \\
7 & 7 & 7 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
0 & 0 & 0 & 1 & 9 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 5 & 6 & 4 & 0 & 0 & 0 & 0 \\
\end{array}
\right]
$$

With the $block\_size = 3$, we can represent $A$ matrix as an $mb \times nb$

$$
A=
\begin{bmatrix}
K & O & O & O \\
L & O & O & M \\
O & O & O & O \\
O & N & O & O
\end{bmatrix}
$$

with the following block matrices:

$$
K=
\begin{bmatrix}
8 & 0 & 2 \\
0 & 3 & 0 \\
7 & 0 & 1
\end{bmatrix}
$$

$$
L=
\begin{bmatrix}
2 & 5 & 0 \\
4 & 0 & 0 \\
7 & 7 & 7 
\end{bmatrix}
$$

$$
M=
\begin{bmatrix}
0 & 0 & 0 \\
5 & 0 & 0 \\
0 & 0 & 0 
\end{bmatrix}
$$

$$
N=
\begin{bmatrix}
1 & 9 & 0 \\
5 & 6 & 4 \\
0 & 0 & 0 
\end{bmatrix}
$$

and an $O$ null matrix:

$$
O=
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 
\end{bmatrix}
$$

The blocks are extended to block size if their size is smaller ($M$ or $N$ blocks).

The BSR representation of the $A$ matrix is the set of ``bsr_val``, ``bsr_row_ptr`` and ``bsr_col_ind``:

``
bsr_val = [ 8, 0, 7, 0, 3, 0, 2, 0, 1,   2, 4, 7, 5, 0, 7, 0, 0, 7,   0, 5, 0, 0, 0, 0, 0, 0, 0,   1, 5, 0, 9, 6, 0, 0, 4, 0 ]
``

array, where contains all elements of the non-zero blocks of the $A$ sparse matrix. The elements are stored block by block in column-major order. Thereby a ``bsr_val`` is a vector with $nnzb \times block\_size\times blocks\_size$ elements, where $nnzb$ is the number of non-zero blocks.

``bsr_row_ptr = { 0, 1, 3, 3, 4 } ``

array of $mb+1$ elements that point to the start of every block row.

``bsr_col_ind = { 0, 0, 3, 1 } ``

array of $nnzb$ elements containing the block column indices.

### rocSPARSE
- `rocsparse_[dscz]bsrmv_ex(...)` is the solver with four different function signatures depending on the type of the input matrix:
   - `d` double-precision real (`double`)
   - `s` single-precision real (`float`)
   - `c` single-precision complex (`rocsparse_float_complex`)
   - `z` double-precision complex (`rocsparse_double_complex`)

- `rocsparse_operation trans`: matrix operation type with the following options:
   - `rocsparse_operation_none`: identity operation: $A' = A$
   - `rocsparse_operation_transpose`: transpose operation: $A' = A^\mathrm{T}$
   - `rocsparse_operation_conjugate_transpose`: Hermitian operation: $A' = A^\mathrm{H}$

   Currently, only `rocsparse_operation_none` is supported.
- `rocsparse_mat_descr`: descriptor of the sparse BSR matrix.
    
- `rocsparse_direction` block storage major direction with the following options:
   - `rocsparse_direction_column`
   - `rocsparse_direction_row`

## Demonstrated API Calls
### rocSPARSE
- `rocsparse_create_handle`
- `rocsparse_create_mat_descr`
- `rocsparse_create_mat_info`
- `rocsparse_dbsrmv_ex`
- `rocsparse_destroy_handle`
- `rocsparse_destroy_mat_descr`
- `rocsparse_destroy_mat_info`
- `rocsparse_direction`
- `rocsparse_direction_column`
- `rocsparse_handle`
- `rocsparse_int`
- `rocsparse_status`
- `rocsparse_mat_descr`
- `rocsparse_mat_info`
- `rocsparse_matrix_type_general`
- `rocsparse_operation`
- `rocsparse_operation_none`

### HIP runtime
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
