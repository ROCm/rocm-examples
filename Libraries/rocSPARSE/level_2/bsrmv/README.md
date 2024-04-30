# rocSPARSE Level 2 BSR Matrix-Vector Multiplication
## Description
This example illustrates the use of the `rocSPARSE` level 2 sparse matrix-vector multiplication using BSR storage format.

The operation calculates the following product:

$$\hat{\mathbf{y}} = \alpha \cdot A' \cdot \mathbf{x} + \beta \cdot \mathbf{y}$$

where

- $\alpha$ and $\beta$ are scalars
- $\mathbf{x}$ and $\mathbf{y}$ are dense vectors
- $A'$ is a sparse matrix in BSR format with `rocsparse_operation` and described below.

## Application flow
1. Set up a sparse matrix in BSR format. Allocate an x and a y vector and set up $\alpha$ and $\beta$ scalars.
2. Set up a handle, a matrix descriptor and a matrix info.
3. Allocate device memory and copy input matrix and vectors from host to device.
4. Compute a sparse matrix multiplication, using BSR storage format.
5. Copy the result vector from device to host.
6. Clear rocSPARSE allocations on device.
7. Clear device arrays.
8. Print result to the standard output.

## Key APIs and Concepts
### BSR Matrix Storage Format
The [Block Compressed Sparse Row (BSR) storage format](https://rocsparse.readthedocs.io/en/latest/usermanual.html#bsr-storage-format) describes a sparse matrix using three arrays. The idea behind this storage format is to split the given sparse matrix into equal sized blocks of dimension `bsr_dim` and store those using the [CSR format](https://rocsparse.readthedocs.io/en/latest/usermanual.html#csr-storage-format). Because the CSR format only stores non-zero elements, the BSR format introduces the concept of __non-zero block__: a block that contains at least one non-zero element. Note that all elements of non-zero blocks are stored, even if some of them are equal to zero.

Therefore, defining
- `mb`: number of rows of blocks
- `nb`: number of columns of blocks
- `nnzb`: number of non-zero blocks
- `bsr_dim`: dimension of each block

we can describe a sparse matrix using the following arrays:
- `bsr_val`: contains the elements of the non-zero blocks of the sparse matrix. The elements are stored block by block in column- or row-major order. That is, it is an array of size `nnzb` $\cdot$ `bsr_dim` $\cdot$ `bsr_dim`.

- `bsr_row_ptr`: given $i \in [0, mb]$
    - if $` 0 \leq i < mb `$, `bsr_row_ptr[i]` stores the index of the first non-zero block in row $i$ of the block matrix
    - if $i = mb$, `bsr_row_ptr[i]` stores `nnzb`.

    This way, row $j \in [0, mb)$ contains the non-zero blocks of indices from `bsr_row_ptr[j]` to `bsr_row_ptr[j+1]-1`. The corresponding values in `bsr_val` can be accessed from `bsr_row_ptr[j] * bsr_dim * bsr_dim` to `(bsr_row_ptr[j+1]-1) * bsr_dim * bsr_dim`.

- `bsr_col_ind`: given $i \in [0, nnzb-1]$, `bsr_col_ind[i]` stores the column of the $i^{th}$ non-zero block in the block matrix.

Note that, for a given $m\times n$ matrix, if the dimensions are not evenly divisible by the block dimension then zeros are padded to the matrix so that $mb$ and $nb$ are the smallest integers greater than or equal to $`\frac{m}{\texttt{bsr\_dim}}`$ and $`\frac{n}{\texttt{bsr\_dim}}`$, respectively.

For instance, consider a sparse matrix as

$$
A=
\left(
\begin{array}{cccc:cccc:cc}
8 & 0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
7 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
2 & 5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 5 \\
7 & 7 & 7 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 9 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 5 & 6 & 4 & 0 & 0 & 0 & 0 \\
\end{array}
\right)
$$

Taking $`\texttt{bsr\_dim} = 4`$, we can represent $A$ as an $mb \times nb$ block matrix

$$
A=
\begin{pmatrix}
A_{00} & O & O \\
A_{10} & O & A_{12} \\
A_{20} & A_{21} & O
\end{pmatrix}
$$

with the following non-zero blocks:

$$
\begin{matrix}
A_{00}=
\begin{pmatrix}
8 & 0 & 2 & 0\\
0 & 3 & 0 & 0\\
7 & 0 & 1 & 0\\
2 & 5 & 0 & 0
\end{pmatrix} &
A_{10}=
\begin{pmatrix}
4 & 0 & 0 & 0 \\
7 & 7 & 7 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} \\
\\
A_{12}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
5 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} &
A_{20}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 5 \\
0 & 0 & 0 & 0
\end{pmatrix} \\
\\
A_{21}=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
9 & 0 & 0 & 0 \\
6 & 4 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix} &
\end{matrix}
$$

and the zero matrix:

$$
O = 0_4=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

Therefore, the BSR representation of $A$, using column-major ordering, is:

```
bsr_val = { 8, 0, 7, 2, 0, 3, 0, 5, 2, 0, 1, 0, 0, 0, 0, 0   // A_{00}
            4, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0   // A_{10}
            0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // A_{12}
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0   // A_{20}
            0, 9, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 } // A_{21}

bsr_row_ptr = { 0, 1, 3, 4 }

bsr_col_ind = { 0, 0, 2, 0, 1 }
```

### rocSPARSE
- `rocsparse_[dscz]bsrmv(...)` performs the sparse matrix-dense vector multiplication $\hat{y}=\alpha \cdot A' x + \beta \cdot y$ using the BSR format. The correct function signature should be chosen based on the datatype of the input matrix:
   - `s` single-precision real (`float`)
   - `d` double-precision real (`double`)
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
- `rocsparse_mat_descr`
- `rocsparse_mat_info`
- `rocsparse_operation`
- `rocsparse_operation_none`

### HIP runtime
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
