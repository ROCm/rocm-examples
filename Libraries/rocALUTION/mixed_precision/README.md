# rocALUTION Mixed-Precision Defect Correction

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using mixed-precision defect correction method.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The mixed-precision defect correction method uses different precision levels for the outer and inner solvers. It typically employs double precision for the outer iteration (to maintain accuracy) and single precision for the inner iteration (to improve performance). This approach can significantly reduce computation time while maintaining solution accuracy.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects (vectors and matrix) in double precision.
5. Read the matrix from MTX format file.
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the mixed-precision defect correction solver:
   - Set up CG solver in single precision as inner solver
   - Configure Multi-Colored ILU preconditioner for the inner solver
   - Set lower tolerance for inner solver (1e-5)
10. Set up the mixed-precision defect correction with double precision outer solver.
11. Build the solver and set verbosity level for output.
12. Print matrix information and start timing measurement.
13. Solve the linear system $Ax = b$.
14. Stop timing measurement and report execution time.
15. Compute and report the L2 norm of the error.
16. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and non-zero count.
  - `rocalution::LocalMatrix::GetN()` and `rocalution::LocalMatrix::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.

- **Mixed-Precision Solver Configuration**:
  - `rocalution::MixedPrecisionDC::SetOperator()`: Associates the linear system matrix with the outer solver.
  - `rocalution::MixedPrecisionDC::Set()`: Sets the inner solver for the defect correction scheme.
  - `rocalution::MixedPrecisionDC::Build()`: Initializes the mixed-precision solver data structures.
  - `rocalution::MixedPrecisionDC::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::MixedPrecisionDC::Solve()`: Executes the mixed-precision solver to find the solution.
  - `rocalution::MixedPrecisionDC::Clear()`: Releases mixed-precision solver resources and memory.

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

- `rocalution::MixedPrecisionDC<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double, rocalution::LocalMatrix<float>, rocalution::LocalVector<float>, float>`
- `rocalution::CG<rocalution::LocalMatrix<float>, rocalution::LocalVector<float>, float>`
- `rocalution::MultiColoredILU<rocalution::LocalMatrix<float>, rocalution::LocalVector<float>, float>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::IterativeLinearSolver::Init`
- `rocalution::MixedPrecisionDC::Set`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::LocalMatrix<float>`
- `rocalution::LocalVector<float>`
