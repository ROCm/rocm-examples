# rocALUTION Block Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using GMRES with a Block preconditioner.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The Block preconditioner divides the matrix into blocks and applies different preconditioners to each block. This approach is particularly useful for matrices with natural block structure or when different regions of the matrix require different preconditioning strategies. This example shows a 2-block configuration with Multi-Colored ILU preconditioners.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Allocate solution, RHS, and error vectors with appropriate dimensions.
6. Initialize the right-hand side vector such that $A \cdot 1 = b$.
7. Set initial solution guess to zero vector.
8. Configure the GMRES solver with Block preconditioner:
   - Set up 2 blocks with equal size distribution
   - Create Multi-Colored ILU preconditioners for each block
   - Configure the Block preconditioner with diagonal solver option
9. Build the solver and set verbosity level for output.
10. Print matrix information and start timing measurement.
11. Solve the linear system $Ax = b$.
12. Stop timing measurement and report execution time.
13. Clear solver resources and free allocated preconditioner objects.
14. Compute and report the L2 norm of the error.
15. Stop the rocALUTION platform.

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

- **Block Preconditioner**:
  - **Block Decomposition**: The matrix is divided into blocks for independent preconditioning.
  - **Flexible Preconditioning**: Each block can use a different preconditioner type.
  - **Diagonal Solver**: Special handling for diagonal blocks within the block structure.
  - **Memory Efficiency**: Block-based approach can reduce memory requirements for large systems.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Block Preconditioner Configuration**:
  - `rocalution::BlockPreconditioner::Set()`: Configures the number of blocks, block sizes, and preconditioners for each block.
  - `rocalution::BlockPreconditioner::SetDiagonalSolver()`: Enables special handling for diagonal blocks.

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
- `rocalution::BlockPreconditioner<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::BlockPreconditioner::Set`
- `rocalution::BlockPreconditioner::SetDiagonalSolver`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::Solver<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
