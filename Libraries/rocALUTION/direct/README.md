# rocALUTION Direct Solver (Matrix Inversion)

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using a direct solver based on matrix inversion.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The direct solver computes the explicit inverse of the matrix $A$ and then multiplies it by the right-hand side vector to obtain the solution: $x = A^{-1}b$. This approach is suitable for small to medium-sized matrices where the cost of computing the inverse is acceptable, and it provides the exact solution (up to numerical precision) in a single step.

## Application flow

1. Parse command line arguments for matrix file path.
2. Initialize the rocALUTION platform.
3. Print rocALUTION platform information.
4. Create rocALUTION objects (vectors and matrix).
5. Read the matrix from MTX format file.
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Configure the direct solver for matrix inversion.
9. Build the solver (compute matrix inverse).
10. Print matrix information and start timing measurement.
11. Solve the linear system $Ax = b$ using the computed inverse.
12. Stop timing measurement and report execution time.
13. Compute and report the L2 norm of the error.
14. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and non-zero count.
  - `rocalution::LocalMatrix::GetN()` and `rocalution::LocalMatrix::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.

- **Direct Solver Configuration**:
  - `rocalution::Inversion::SetOperator()`: Associates the linear system matrix with the direct solver.
  - `rocalution::Inversion::Build()`: Computes the inverse of the matrix for direct solving.
  - `rocalution::Inversion::Solve()`: Solves the linear system using the pre-computed matrix inverse.
  - `rocalution::Inversion::Clear()`: Releases direct solver resources and memory.

- **Performance Measurement**:
  - `rocalution::rocalution_time()`: Returns high-resolution timer value for performance measurement.

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`
- `rocalution::info_rocalution`
- `rocalution::rocalution_time`

### Matrix Operations

- `rocalution::LocalMatrix::ReadFileMTX`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::GetM`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::ScaleAdd`
- `rocalution::LocalVector::Norm`

### Solver Classes

- `rocalution::Inversion<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::Inversion::SetOperator`
- `rocalution::Inversion::Build`
- `rocalution::Inversion::Solve`
- `rocalution::Inversion::Clear`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
