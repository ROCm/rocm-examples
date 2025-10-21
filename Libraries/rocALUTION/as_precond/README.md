# rocALUTION Additive Schwarz Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using GMRES with an Additive Schwarz preconditioner.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The Additive Schwarz preconditioner is a domain decomposition method that divides the problem into overlapping subdomains, solves local problems on each subdomain, and combines the results. This example shows a two-level Additive Schwarz preconditioner where the second level uses Multi-Colored ILU preconditioners.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Allocate solution, RHS, and error vectors with appropriate dimensions.
6. Initialize the right-hand side vector such that $A \cdot 1 = b$.
7. Set initial solution guess to zero vector.
8. Configure the GMRES solver with Additive Schwarz preconditioner.
9. Set up second-level preconditioners:
   - Create 2 Multi-Colored ILU preconditioners for the second level
   - Configure the Additive Schwarz preconditioner with 2 subdomains and 4 overlap size
10. Build the solver and move all objects to the accelerator.
11. Print matrix information and start timing measurement.
12. Solve the linear system $Ax = b$.
13. Stop timing measurement and report execution time.
14. Clear solver resources and free allocated preconditioner objects.
15. Compute and report the L2 norm of the error.
16. Stop the rocALUTION platform.

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

- **Additive Schwarz Preconditioner**:
  - **Domain Decomposition**: The matrix is conceptually divided into overlapping subdomains.
  - **Two-Level Preconditioning**: Local problems are solved on subdomains (first level), and their results are combined using a coarse grid correction (second level).
  - **Overlap**: Subdomains overlap to ensure good convergence properties.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Additive Schwarz Configuration**:
  - `rocalution::AS::Set()`: Configures the number of subdomains, overlap size, and second-level preconditioners.

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

- `rocalution::GMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::AS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::AS::Set`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::Solver<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
