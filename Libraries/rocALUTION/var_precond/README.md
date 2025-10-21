# rocALUTION Variable Preconditioner

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using Flexible Generalized Minimal Residual with a Variable preconditioner.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The Variable preconditioner allows switching between different preconditioners during the solving process. This approach is useful when different preconditioners are effective at different stages of the iteration or when the problem structure changes during solving. This example shows how to configure a variable preconditioner that can switch between Multi-Colored SGS, Multi-Colored ILU, and standard ILU preconditioners.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Move all objects to the accelerator (GPU).
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the FGMRES solver with Variable preconditioner:
   - Set up three different preconditioners (Multi-Colored SGS, Multi-Colored ILU, ILU)
   - Configure the Variable preconditioner with the three preconditioners
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

- **Variable Preconditioner**:
  - **Dynamic Switching**: Allows changing preconditioners during the solving process.
  - **Flexibility**: Different preconditioners can be used for different iteration phases.
  - **Adaptive Strategy**: The solver can adapt its preconditioning strategy based on convergence behavior.
  - **Performance Optimization**: Switching between preconditioners can improve overall convergence rate.

- **FGMRES with Variable Preconditioning**:
  - **Flexible Framework**: FGMRES naturally accommodates changing preconditioners.
  - **Krylov Subspace**: Maintains orthogonality even when preconditioners change.
  - **Restart Strategy**: Variable preconditioning can be combined with restart strategies.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Variable Preconditioner Configuration**:
  - `rocalution::VariablePreconditioner::SetPreconditioner()`: Configures the variable preconditioner with multiple preconditioner options.

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

- `rocalution::FGMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::VariablePreconditioner<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredSGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::ILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::VariablePreconditioner::SetPreconditioner`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::Solver<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
