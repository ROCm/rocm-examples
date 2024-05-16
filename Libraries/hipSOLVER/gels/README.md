# hipSOLVER linear least-squares

## Description

This example illustrates the use of hipSOLVER's linear least-squares solver, `gels`. The `gels` functions solve an overdetermined (or underdetermined) linear system defined by an $m$-by-$n$ matrix $A$, and a corresponding matrix $B$, using the QR factorization computed by `geqrf` (or the LQ factorization computed by `gelqf`). The problem solved by this function is of the form $A\times X=B$.

If $m\geq n$, the system is overdetermined and a least-squares solution approximating $X$ is found by minimizing $||B−A\times X||$ (or $||B−A^\prime\times X||$). If $m \text{ \textless}\ n$, the system is underdetermined and a unique solution for X is chosen such that $||X||$ is minimal.

This example shows how $A\times X = B$ is solved for $X$, where $X$ is an $m$-by-$1$ matrix. The result is validated by calculating $A\times X$ for the found result, and comparing that with $B$.

### Application flow

1. Parse the user inputs, declare several constants for the sizes of the matrices.
2. Allocate the input- and output matrices on the host and device, initialize the input data.
3. Create a hipSOLVER handle.
4. Query the size of the working space of the `gels` function and allocate the required amount of device memory.
5. Call the `gels` function to solve the linear least squares problem: $A\times X=B$.
6. Copy the device result back to the host.
7. Print the status value of the `gels` function.
8. Free device resources and the hipSOLVER handle.
9. Validate that the result found is correct by calculating $A\times X$, and print the result.

### Command line interface

The application provides the following optional command line arguments:

- `--n <n>`. Number of rows of input matrix $A$, the default value is `3`.
- `--m <m>`. Number of columns of input matrix $A$, the default value is `2`.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.

- `hipsolver(SS|DD|CC|ZZ)gels` solves the system of linear equations defined by $A\times X=B$, where $A$ is an `m`-by-`n` matrix, $X$ is an `n`-by-`nrhs` matrix, and $B$ is an `m`-by-`nrhs` matrix. Depending on the character matched in `(SS|DD|CC|ZZ)`, the solution can be obtained with different precisions:

  - `S` (single-precision: `float`).
  - `D` (double-precision: `double`).
  - `C` (single-precision complex: `hipFloatComplex`).
  - `Z` (double-precision complex: `hipDoubleComplex`).

  The `gels` function also requires the specification of the _leading dimension_ of all matrices. The leading dimension specifies the number of elements between the beginnings of successive matrix vectors. In other fields, this may be referred to as the _stride_. This concept allows the matrix used in the `gels` function to be a sub-matrix of a larger one. Since hipSOLVER matrices are stored in column-major order, the leading dimension must be greater than or equal to the number of rows of the matrix.

- `hipsolver(SS|DD|CC|ZZ)gels_bufferSize` allows to obtain the size needed for the working space for the `hipsolver(SS|DD|CC|ZZ)gels` function.

## Used API surface

### hipSOLVER

- `hipsolverDDgels`
- `hipsolverDDgels_bufferSize`
- `hipsolverHandle_t`
- `hipsolverCreate`
- `hipsolverDestroy`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
