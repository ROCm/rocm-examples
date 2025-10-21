# rocALUTION Sparse Matrix-Vector Multiplication (SpMV)

## Description

This example demonstrates the use of the `rocALUTION` library for performing sparse matrix-vector multiplication operations.

The operation computes:

$y = Ax$

where

- $A$ is a sparse matrix
- $x$ is the input vector
- $y$ is the output vector

This example focuses on the fundamental sparse matrix-vector multiplication (SpMV) operation, which is a key building block for many iterative solvers. It demonstrates matrix format conversion, memory management, and both host and device execution.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects (vectors and matrix).
5. Read the sparse matrix from MTX format file.
6. Print matrix information and allocate vectors with appropriate dimensions.
7. Print vector information and initialize the input vector to ones.
8. Perform sparse matrix-vector multiplication on the host.
9. Compute and print the dot product of the result.
10. Convert the matrix to ELL format for optimized storage.
11. Print updated matrix information after format conversion.
12. Move all objects to the accelerator (GPU).
13. Print matrix information after moving to device.
14. Re-initialize the input vector and perform SpMV on the device.
15. Compute and print the dot product from the device computation.
16. Stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a sparse matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::ConvertToELL()`: Converts the matrix to ELL (Ellpack-Itpack) format for optimized GPU execution.
  - `rocalution::LocalMatrix::Apply()`: Performs sparse matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints detailed matrix information including format, dimensions, and non-zero count.
  - `rocalution::LocalMatrix::GetN()` and `rocalution::LocalMatrix::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Dot()`: Computes the dot product of two vectors.
  - `rocalution::LocalVector::Info()`: Prints vector information including size and location.

- **Memory Management**:
  - `rocalution::LocalMatrix::MoveToAccelerator()`: Transfers matrix data from host to device memory.
  - `rocalution::LocalVector::MoveToAccelerator()`: Transfers vector data from host to device memory.

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`
- `rocalution::info_rocalution`
- `rocalution::set_omp_threads_rocalution`

### Matrix Operations

- `rocalution::LocalMatrix::ReadFileMTX`
- `rocalution::LocalMatrix::ConvertToELL`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::GetM`
- `rocalution::LocalMatrix::MoveToAccelerator`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::Dot`
- `rocalution::LocalVector::Info`
- `rocalution::LocalVector::MoveToAccelerator`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
