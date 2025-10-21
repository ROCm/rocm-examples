# rocALUTION Flexible Generalized Minimal Residual (FGMRES)

## Description

This example demonstrates the use of the `rocALUTION` library for solving nonsymmetric linear systems using the Flexible Generalized Minimal Residual method.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a nonsymmetric matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The Flexible GMRES method is an extension of GMRES that allows the preconditioner to change at each iteration. This is particularly useful when using variable preconditioners or when the preconditioner itself is an iterative method. FGMRES maintains orthogonality in the Krylov subspace while accommodating flexible preconditioning.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects (vectors and matrix).
5. Read the matrix from MTX format file.
6. Move all objects to the accelerator (GPU).
7. Allocate solution, RHS, and error vectors with appropriate dimensions.
8. Initialize the right-hand side vector such that $A \cdot 1 = b$.
9. Set initial solution guess to zero vector.
10. Configure the FGMRES solver with Multi-Colored Gauss-Seidel preconditioner.
11. Set FGMRES basis size to 15 for the Krylov subspace dimension.
12. Build the solver and set verbosity level for output.
13. Print matrix information and start timing measurement.
14. Solve the linear system $Ax = b$.
15. Stop timing measurement and report execution time.
16. Compute and report the L2 norm of the error.
17. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::MoveToAccelerator()`: Transfers matrix data from host to device memory.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and non-zero count.
  - `rocalution::LocalMatrix::GetN()` and `rocalution::LocalMatrix::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **FGMRES-Specific Configuration**:
  - `rocalution::FGMRES::SetBasisSize()`: Sets the dimension of the Krylov subspace basis for FGMRES restart.

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
- `rocalution::LocalMatrix::MoveToAccelerator`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::GetM`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::Zeros`
- `rocalution::LocalVector::ScaleAdd`
- `rocalution::LocalVector::Norm`

### Solver Classes

- `rocalution::FGMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::FGMRES::SetBasisSize`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
