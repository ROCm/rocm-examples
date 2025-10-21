# rocALUTION Power Method and Chebyshev Iteration

## Description

This example demonstrates the use of the `rocALUTION` library for eigenvalue approximation using the Power Method and solving linear systems using Chebyshev iteration.

The example covers three main operations:

1. **Power Method for Eigenvalue Approximation**: Estimates the maximum and minimum eigenvalues of a matrix
2. **Chebyshev Iteration**: Solves linear systems using Chebyshev polynomial iteration
3. **CG with Chebyshev Preconditioner**: Combines Conjugate Gradient with Approximate Inverse Chebyshev preconditioning

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. **Gershgorin Circle Theorem**: Compute initial eigenvalue bounds using Gershgorin approximation.
6. Move objects to the accelerator and allocate vectors for power iteration.
7. **Power Method for λ_max**:
    - Initialize vector to ones and iterate matrix-vector multiplications
    - Normalize vectors and compute Rayleigh quotient for maximum eigenvalue
8. **Power Method for λ_min**:
    - Modify matrix by subtracting λ_max from diagonal
    - Apply power method to find minimum eigenvalue of shifted matrix
    - Restore original matrix and compute actual minimum eigenvalue
9. **Chebyshev Iteration**:
    - Set up Chebyshev solver with computed eigenvalue bounds
    - Solve linear system using Chebyshev polynomial iteration
10. **CG with AIChebyshev Preconditioner**:
    - Set up Approximate Inverse Chebyshev preconditioner
    - Solve linear system using CG with Chebyshev preconditioning
11. Compute and report errors for both solving approaches.
12. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::Gershgorin()`: Computes eigenvalue bounds using Gershgorin Circle Theorem.
  - `rocalution::LocalMatrix::AddScalarDiagonal()`: Adds a scalar value to the matrix diagonal.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and format.

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Scale()`: Scales vector elements by a scalar value.
  - `rocalution::LocalVector::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.
  - `rocalution::LocalVector::Dot()`: Computes the dot product of two vectors.
  - `rocalution::LocalVector::CloneBackend()`: Clones the backend (host/device) configuration from another object.
  - `rocalution::LocalVector::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.

- **Eigenvalue Approximation**:
  - **Power Method**: Iterative algorithm to find extreme eigenvalues through repeated matrix-vector multiplication and normalization.
  - **Rayleigh Quotient**: Computed as $\lambda = \frac{x^T A x}{x^T x}$ to estimate eigenvalues.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Chebyshev-Specific Configuration**:
  - `rocalution::Chebyshev::Set()`: Sets eigenvalue bounds for Chebyshev polynomial iteration.
  - `rocalution::AIChebyshev::Set()`: Configures Approximate Inverse Chebyshev preconditioner with polynomial degree and eigenvalue bounds.

- **Performance Measurement**:
  - `rocalution::rocalution_time()`: Returns high-resolution timer value for performance measurement.

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`
- `rocalution::info_rocalution`
- `rocalution::set_omp_threads_rocalution`
- `rocalution::rocalution_time`

### Matrix Operations

- `rocalution::LocalMatrix::ReadFileMTX`
- `rocalution::LocalMatrix::Gershgorin`
- `rocalution::LocalMatrix::AddScalarDiagonal`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetM`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::MoveToAccelerator`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::Scale`
- `rocalution::LocalVector::Norm`
- `rocalution::LocalVector::Dot`
- `rocalution::LocalVector::CloneBackend`
- `rocalution::LocalVector::Zeros`
- `rocalution::LocalVector::ScaleAdd`
- `rocalution::LocalVector::MoveToAccelerator`

### Solver Classes

- `rocalution::Chebyshev<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::CG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::AIChebyshev<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::Chebyshev::Set`
- `rocalution::AIChebyshev::Set`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
