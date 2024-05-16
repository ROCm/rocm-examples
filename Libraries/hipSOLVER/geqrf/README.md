# hipSOLVER QR Factorization Example

## Description

This example illustrates the use of hipSOLVER to compute the QR factorization of a matrix $A$. The [QR factorization](https://en.wikipedia.org/wiki/QR_decomposition) of a $m \times n$ matrix $A$ computes the unitary matrix $Q$ and upper triangular matrix $R$, such that $A = QR$. The QR factorization is calculated using householder transformations.

- $Q$ is an $m \times m$ unitary matrix, i.e. $Q^{-1} = Q^H$
- $R$ is an $m \times n$ upper (or right) triangular matrix, i.e. all entries below the diagonal are zero.

In general, a rectangular matrix $A$ with $m > n$ and $rank(A) = n$ can be factorized as the product of an $m \times m$ unitary matrix $Q$ and an $m \times n$ upper triangular matrix $R$. In that case $Q$ can be partitioned into $\begin{pmatrix} Q_1 & Q_2 \end{pmatrix}$, with $Q_1$ being a $m \times n$, and $Q_2$ being a $m \times (m - n)$ matrix with orthonormal columns each, and $R$ can be partitioned into $\begin{pmatrix} R_1 \\ 0 \end{pmatrix}$, where $R_1$ is an $n \times n$ upper triangular matrix and $0$ is a $(m - n) \times n$ zero matrix. Thereby $A = Q R = Q_1 R_1$.

In the general case hipSOLVER calculates $Q_1$ and $R_1$.

The calculated solution is verified by computing the root mean square of the elements in $Q^H Q - I$, which should result in a zero matrix, to check whether $Q$ is actually orthogonal.

### Application flow

1. Declare and initialize variables for the in- and output matrix.
2. Initialize the matrix on the host.
3. Allocate device memory and copy the matrix to the device.
4. Create a hipSOLVER handle and set up the working space needed for the QR factorization functions.
5. Compute the first step of the QR factorization using `hipsolverDgeqrf`.
6. Compute the matrix Q with the householder vectors stored in d_A and the scaling factors in d_tau using `hipsolverDorgqr`.
7. Validate the result by calculating the root mean square of the elements in $Q^T Q - I$ using hipBLAS.
8. Free device memory and handles.

## Key APIs and Concepts

### hipSOLVER

- hipSOLVER is initialized by calling `hipsolverCreate(hipsolverHandle_t*)` and it is terminated by calling `hipsolverDestroy(hipsolverHandle_t)`.

- `hipsolver[SDCZ]geqrf` computes the QR factorization of a $m \times n$ matrix $A$. The results of $Q$ and $R$ are stored in place of $A$. The orthogonal matrix $Q$ is not explicitly calculated, it is stored using householder vectors, which can be used to explicitly calculate $Q$ with `hipsolver[SDCZ]orgqr`. Depending on the character matched in `[SDCZ]`, the QR factorization can be obtained with different precisions:

  - `S` (single-precision: `float`)
  - `D` (double-precision: `double`)
  - `C` (single-precision complex: `hipFloatComplex`)
  - `Z` (double-precision complex: `hipDoubleComplex`).

  In this example the double-precision variant `hipsolverDgeqrf` is used.

  Its input parameters are:

  - `hipsolverHandle_t handle`
  - `int m` number of rows of $A$
  - `int n` number of columns of $A$
  - `double *A` pointer to matrix $A$
  - `int lda` leading dimension of matrix $A$
  - `double *tau` vector that stores the scaling factors for the householder vectors.
  - `double *work` memory for working space used by the function
  - `int lwork` size of working space
  - `int *devInfo` status report of the function. The QR factorization is successful if the value pointed to by devInfo is 0. When using cuSOLVER as backend, if the value is a negative integer $-i$, then the i-th parameter of `hipsolverDgeqrf` is wrong.

  The return type is `hipsolverStatus_t`.

- `hipsolver[SDCZ]geqrf_bufferSize` calculates the required size of the working space for `hipsolver[SDCZ]geqrf`. The used type has to match the actual solver function.

  The input parameters for `hipsolverDgeqrf_bufferSize` are:

  - `hipsolverHandle_t handle`
  - `int m` number of rows of $A$
  - `int n` number of columns of $A$
  - `double *A` pointer to matrix $A$
  - `int lda` leading dimension of matrix $A$
  - `int *lwork` returns the size of the working space required

  The return type is `hipsolverStatus_t`.

- `hipsolver[SD]orgqr` computes the orthogonal matrix $Q$ from the householder vectors, as stored in $A$, and the corresponding scaling factors as stored in tau, both as returned by `hipsolver[SD]geqrf`.

  In the case of complex matrices, the function `hipsolver[CZ]ungqr` has to be used.
  In this example the double-precision variant `hipsolverDorgqr` is used.

  Its input parameters are:

  - `hipsolverHandle_t handle`
  - `int m` number of rows of matrix $Q$
  - `int n` number of columns of matrix $Q$ ($m \geq n \gt 0$)
  - `int k` number of elementary reflections whose product defines the matrix $Q$ ($n \geq k \geq 0$)
  - `double *A` matrix containing the householder vectors
  - `int lda` leading dimension of $A$
  - `double *tau` vector that stores the scaling factors for the householder vectors
  - `double *work` memory for working space used by the function
  - `int lwork` size of working space
  - `int *devInfo` status report of the function. The computation of $Q$ is successful if the value pointed to by devInfo is 0. When using cuSOLVER as backend, if the value is a negative integer $-i$, then the i-th parameter of `hipsolverDorgqr` is wrong.

  The return type is `hipsolverStatus_t`.

- `hipsolver[SD]orgqr_bufferSize` calculates the required size of the working space for `hipsolver[SD]orgqr`. The used type has to match the actual solver function.

  The input parameters for `hipsolverDorgqr_bufferSize` are:

  - `hipsolverHandle_t handle`
  - `int m` number of rows of matrix $Q$
  - `int n` number of columns of matrix $Q$
  - `int k` number of elementary reflection
  - `double *A` matrix containing the householder vectors
  - `int lda` leading dimension of $A$
  - `double *tau` vector that stores the scaling factors for the householder vectors
  - `int *lwork` returns the size of the working space required

  The return type is `hipsolverStatus_t`.

### hipBLAS

hipBLAS is used to validate the solution. To verify that $Q$ is orthogonal the solution $Q^T Q - I$ is computed using `hipblasDgemm` and the root mean square of the elements of that result is calculated using `hipblasDnrm2`. `hipblasDgemm` is showcased in the [gemm_strided_batched example](/Libraries/hipBLAS/gemm_strided_batched/).

`hipblasDnrm2` calculates the euclidean norm of a vector. In this example the root mean square of the elements in a matrix is calculated by pretending it to be a vector and calculating its euclidean norm, then dividing it by the number of elements in the matrix.

Its input parameters are:

- `hipblasHandle_t handle`
- `int n` number of elements in x
- `double *x` device pointer storing vector x
- `int incx` stride between consecutive elements of x
- `double *result` resulting norm
- The `hipblasPointerMode_t` type controls whether scalar parameters must be allocated on the host (`HIPBLAS_POINTER_MODE_HOST`) or on the device (`HIPBLAS_POINTER_MODE_DEVICE`). It is set by using `hipblasSetPointerMode`.

## Used API surface

### hipSOLVER

- `hipsolverCreate`
- `hipsolverDestroy`
- `hipsolverDgeqrf`
- `hipsolverDgeqrf_bufferSize`
- `hipsolverDorgqr`
- `hipsolverDorgqr_bufferSize`
- `hipsolverHandle_t`

### hipBLAS

- `hipblasCreate`
- `hipblasDestroy`
- `hipblasDgemm`
- `hipblasDnrm2`
- `hipblasHandle_t`
- `HIPBLAS_OP_N`
- `HIPBLAS_OP_T`
- `HIPBLAS_POINTER_MODE_HOST`
- `hipblasSetPointerMode`

### HIP runtime

- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
