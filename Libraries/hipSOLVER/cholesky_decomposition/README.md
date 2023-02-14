# hipSOLVER Cholesky Decomposition and linear system solver

## Description
This example illustrates the functionality to perform Cholesky decomposition, POTRF, and to solve a linear system using the resulting Cholesky factor, POTRS. The POTRF functions decompose a Hermitian positive-definite matrix $A$ into $L\cdot L^H$ (or $U^H\cdot U$), where $L$ and $U$ are a lower- and upper-triangular matrix, respectively. The POTRS functions solve a linear system $A\times X=B$ for $X$.

### Application flow
1. Declare several constants for the sizes of the matrices.
2. Allocate the input- and output-matrices on the host and device, initialize the input data. Matrix $A_0$ is not
   Hermitian positive semi-definite, matrix $A_1$ is Hermitian positive semi-definite.
3. Create a hipSOLVER handle.
4. Query the size of the working space of the POTRF function and allocate the required amount of device memory.
5. Call the POTRF function to decompose $A_0$ and assert that it failed since $A_0$ does not meet the requirements.
6. Call the POTRF function to decompose $A_1$ and assert that it succeeds.
7. Call the POTRS function to solve the system $A_1\times X=B$.
8. Copy the device result back to the host.
9. Free device resources and the hipSOLVER handle.
10. Validate that the result found is correct by calculating $A_1\times X$, and print the result.

## Key APIs and Concepts
### hipSOLVER
- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.
- `hipsolver(SnS|DnD|CnC|ZnZ)potrf` performs Cholesky decomposition on Hermitian positive semi-definite matrix $A$. Depending on the character matched in `(SnS|DnD|CnC|ZnZ)`, the solution can be obtained with different precisions:
    - `S` (single-precision: `float`).
    - `D` (double-precision: `double`).
    - `C` (single-precision complex: `hipFloatComplex`).
    - `Z` (double-precision complex: `hipDoubleComplex`).
- `hipsolver(SnS|DnD|CnC|ZnZ)potrf_bufferSize` obtains the size needed for the working space for the `hipsolver(SnS|DnD|CnC|ZnZ)potrf` function.
- `hipsolver(SnS|DnD|CnC|ZnZ)potrs` solves the system of linear equations defined by $A\times X=B$, where $A$ is a Cholesky-decomposed Hermitian positive semi-definite `n`-by-`n` matrix, $X$ is an `n`-by-`nrhs` matrix, and $B$ is an `n`-by-`nrhs` matrix.
- The POTRF and POTRS functions require the specification of a `hipsolverFillMode_t`, which indicates which triangular part of the matrix is processed and replaced by the functions. The legal values are `HIPSOLVER_FILL_MODE_LOWER` and `HIPSOLVER_FILL_MODE_UPPER`.
- The POTRF and POTRS functions also require the specification of the _leading dimension_ of all matrices. The leading dimension specifies the number of elements between the beginnings of successive matrix vectors. In other fields, this may be referred to as the _stride_. This concept allows the matrix used in the POTRF and POTRS functions to be a sub-matrix of a larger one. Since hipSOLVER matrices are stored in column-major order, the leading dimension must be greater than or equal to the number of rows of the matrix.

## Used API surface
### hipSOLVER
- `HIPSOLVER_FILL_MODE_LOWER`
- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverDnDpotrf`
- `hipsolverDnDpotrf_bufferSize`
- `hipsolverDnDpotrs`
- `hipsolverFillMode_t`
- `hipsolverHandle_t`

### HIP runtime
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
