# hipSOLVER Cholesky Decomposition and linear system solver

## Description

This example illustrates the functionality to perform Cholesky decomposition, `potrf`, and to solve a linear system using the resulting Cholesky factor, `potrs`. The `potrf` functions decompose a Hermitian positive-definite matrix $A$ into $L\cdot L^H$ (or $U^H\cdot U$), where $L$ and $U$ are a lower- and upper-triangular matrix, respectively. The `potrs` functions solve a linear system $A\times X=B$ for $X$.

### Application flow

1. Declare several constants for the sizes of the matrices.
2. Allocate the input- and output-matrices on the host and device, initialize the input data. Matrix $A_0$ is not Hermitian positive semi-definite, matrix $A_1$ is Hermitian positive semi-definite.
3. Create a hipSOLVER handle.
4. Query the size of the working space of the `potrf` and `potrs` functions and allocate the required amount of device memory.
5. Call the `potrf` function to decompose $A_0$ and assert that it failed since $A_0$ does not meet the requirements.
6. Call the `potrf` function to decompose $A_1$ and assert that it succeeds.
7. Call the `potrs` function to solve the system $A_1\times X=B$.
8. Copy the device result back to the host.
9. Free device resources and the hipSOLVER handle.
10. Validate that the result found is correct by calculating $A_1\times X$, and print the result.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.

- `hipsolver[SDCZ]potrf` performs Cholesky decomposition on Hermitian positive semi-definite matrix $A$. The correct function signature should be chosen based on the datatype of the input matrix:

  - `S` (single-precision: `float`).
  - `D` (double-precision: `double`).
  - `C` (single-precision complex: `hipFloatComplex`).
  - `Z` (double-precision complex: `hipDoubleComplex`).

- `hipsolver[SDCZ]potrf_bufferSize` obtains the size needed for the working space for the `hipsolver[SDCZ]potrf` function.

- `hipsolver[SDCZ]potrs` solves the system of linear equations defined by $A\times X=B$, where $A$ is a Cholesky-decomposed Hermitian positive semi-definite `n`-by-`n` matrix, $X$ is an `n`-by-`nrhs` matrix, and $B$ is an `n`-by-`nrhs` matrix.

- The `potrf` and `potrs` functions require the specification of a `hipsolverFillMode_t`, which indicates which triangular part of the matrix is processed and replaced by the functions. The legal values are `HIPSOLVER_FILL_MODE_LOWER` and `HIPSOLVER_FILL_MODE_UPPER`.

- The `potrf` and `potrs` functions also require the specification of the _leading dimension_ of all matrices. The leading dimension specifies the number of elements between the beginnings of successive matrix vectors. In other fields, this may be referred to as the _stride_. This concept allows the matrix used in the `potrf` and `potrs` functions to be a sub-matrix of a larger one. Since hipSOLVER matrices are stored in column-major order, the leading dimension must be greater than or equal to the number of rows of the matrix.

## Used API surface

### hipSOLVER

- `HIPSOLVER_FILL_MODE_LOWER`
- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverDpotrf`
- `hipsolverDpotrf_bufferSize`
- `hipsolverDpotrs`
- `hipsolverFillMode_t`
- `hipsolverHandle_t`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
