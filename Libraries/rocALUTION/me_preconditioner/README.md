# rocALUTION Multi-Elimination Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving symmetric positive definite linear systems using the Conjugate Gradient method with Multi-Elimination preconditioning.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a symmetric positive definite matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how to combine the CG solver with Multi-Elimination preconditioning for optimal performance on systems with suitable structure. The Multi-Elimination preconditioner uses a block elimination approach with Multi-Colored ILU as the last block preconditioner, providing efficient preconditioning for matrices that can be effectively partitioned.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Move all objects to the accelerator (GPU).
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the CG solver with Multi-Elimination preconditioner:
   - Set up Multi-Colored ILU as the last block preconditioner
   - Configure the Multi-Elimination preconditioner with the ILU preconditioner, 2 levels, and 0.4 damping factor
10. Build the solver and set verbosity level for output.
11. Print matrix information and start timing measurement.
12. Solve the linear system $Ax = b$.
13. Stop timing measurement and report execution time.
14. Compute and report the L2 norm of the error.
15. Clear solver resources and stop the rocALUTION platform.

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

- **Multi-Elimination Preconditioner**:
  - `rocalution::MultiElimination::Set()`: Configures the multi-elimination preconditioner with block preconditioner, levels, and damping factor.
  - **Block Elimination**: Divides the matrix into blocks and eliminates variables in a systematic way.
  - **Damping Factor**: Controls the stability and convergence rate of the elimination process.
  - **Last Block Preconditioner**: Special preconditioning for the final block after elimination.

- **Multi-Colored ILU Configuration**:
  - `rocalution::MultiColoredILU::Set()`: Configures the Multi-Colored ILU preconditioner parameters.

- **Multi-Elimination Concepts**:
  - **Block Structure**: Exploits natural block structure in the matrix for efficient elimination.
  - **Recursive Elimination**: Eliminates variables block by block, reducing system size progressively.
  - **Parallel Efficiency**: Multi-colored approach enables parallel execution of elimination steps.

- **CG with Multi-Elimination Preconditioner Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the CG solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the Multi-Elimination preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes CG solver data structures and Multi-Elimination preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the CG solver with Multi-Elimination preconditioning to find the solution.
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
- `rocalution::MultiElimination<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::MultiElimination::Set`
- `rocalution::MultiColoredILU::Set`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
