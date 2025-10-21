# rocALUTION CG with SAAMG Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving symmetric positive definite linear systems using the Conjugate Gradient method with Smoothed Aggregation Algebraic Multigrid preconditioning.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a symmetric positive definite matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how to combine the CG solver with SAAMG preconditioning for optimal performance on large symmetric positive definite systems. The SAAMG preconditioner automatically constructs a multigrid hierarchy to accelerate convergence, while the CG solver provides the outer iteration framework.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and move them to the accelerator.
5. Read the matrix from MTX format file.
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the CG solver with SAAMG preconditioner.
10. Build the solver and set the AMG preconditioner to compute 2 coarsest levels on the host.
11. Print matrix information and measure build time.
12. Initialize solver tolerances and set verbosity level for output.
13. Solve the linear system $Ax = b$.
14. Stop timing measurement and report execution time.
15. Compute and report the L2 norm of the error.
16. Clear solver resources and stop the rocALUTION platform.

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
  - `rocalution::LocalVector::MoveToAccelerator()`: Transfers vector data from host to device memory.

- **CG with AMG Preconditioner Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the CG solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the SAAMG preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes CG solver data structures and AMG preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Init()`: Sets solver tolerances and iteration limits.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the CG solver with AMG preconditioning to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **SAAMG Configuration**:
  - `rocalution::SAAMG::Verbose()`: Sets the verbosity level for AMG preconditioner progress output.
  - `rocalution::SAAMG::SetHostLevels()`: Sets the number of coarsest levels to compute on the host CPU.

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
- `rocalution::LocalVector::MoveToAccelerator`

### Solver Classes

- `rocalution::CG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::SAAMG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Init`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::SAAMG::Verbose`
- `rocalution::SAAMG::SetHostLevels`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
