# rocALUTION CG with Ruge-Stueben AMG Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving symmetric positive definite linear systems using the Conjugate Gradient method with Ruge-Stueben Algebraic Multigrid preconditioning.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a symmetric positive definite matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how to combine the CG solver with Ruge-Stueben AMG preconditioning for optimal performance on large symmetric positive definite systems. The Ruge-Stueben AMG uses classical coarsening with PMIS strategy and Extended Prolongated Interpolation for robust multigrid hierarchy construction.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and move them to the accelerator.
5. Read the matrix from MTX format file.
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the CG solver with Ruge-Stueben AMG preconditioner:
   - Set coarsening strategy to PMIS (Parallel Modified Independent Set)
   - Set interpolation type to Extended Prolongated Interpolation (ExtPI)
   - Set coarsest level to 20 unknowns
   - Disable operator complexity limit for interpolation
   - Configure 2 coarsest levels to compute on host
10. Build the solver and measure build time.
11. Print matrix information and initialize solver tolerances.
12. Set verbosity level for output and start timing measurement.
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

- **Ruge-Stueben AMG Configuration**:
  - `rocalution::RugeStuebenAMG::SetCoarseningStrategy()`: Sets the coarsening strategy (PMIS).
  - `rocalution::RugeStuebenAMG::SetInterpolationType()`: Sets the interpolation type (ExtPI).
  - `rocalution::RugeStuebenAMG::SetCoarsestLevel()`: Sets the maximum number of unknowns on the coarsest level.
  - `rocalution::RugeStuebenAMG::SetInterpolationFF1Limit()`: Controls operator complexity in interpolation.
  - `rocalution::RugeStuebenAMG::Verbose()`: Sets the verbosity level for AMG preconditioner progress output.
  - `rocalution::RugeStuebenAMG::SetHostLevels()`: Sets the number of coarsest levels to compute on the host CPU.

- **Ruge-Stueben AMG**:
  - **Classical Coarsening**: Uses traditional Ruge-Stueben coarsening algorithm.
  - **PMIS Strategy**: Parallel Modified Independent Set for optimal parallel performance.
  - **Extended Prolongated Interpolation**: Advanced interpolation technique for better convergence.
  - **Robustness**: Classical AMG approach provides reliable convergence for many problem types.

- **CG with AMG Preconditioner Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the CG solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the AMG preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes CG solver data structures and AMG preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Init()`: Sets solver tolerances and iteration limits.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the CG solver with AMG preconditioning to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

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
- `rocalution::RugeStuebenAMG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Init`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::RugeStuebenAMG::SetCoarseningStrategy`
- `rocalution::RugeStuebenAMG::SetInterpolationType`
- `rocalution::RugeStuebenAMG::SetCoarsestLevel`
- `rocalution::RugeStuebenAMG::SetInterpolationFF1Limit`
- `rocalution::RugeStuebenAMG::Verbose`
- `rocalution::RugeStuebenAMG::SetHostLevels`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::CoarseningStrategy`
- `rocalution::InterpolationType`
