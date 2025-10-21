# rocALUTION Iterative Triangular Solve

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using the Fixed-Point iteration method with Iterative ILU0 preconditioning.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how to use the Iterative ILU0 preconditioner, which applies an iterative algorithm to solve the triangular systems that arise during ILU factorization. The iterative triangular solve uses a configurable solver descriptor and provides detailed convergence history, making it suitable for analyzing the behavior of iterative preconditioning methods.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Move all objects to the accelerator (GPU).
6. Allocate solution, RHS, and error vectors with appropriate dimensions.
7. Initialize the right-hand side vector such that $A \cdot 1 = b$.
8. Set initial solution guess to zero vector.
9. Configure the Fixed-Point solver with Iterative ILU0 preconditioner:
   - Set tolerance to 1e-8 and maximum iterations to 50
   - Enable comprehensive options (verbose, stopping criteria, norm computation, convergence history)
   - Set algorithm to SyncSplit for parallel execution
   - Configure iterative triangular solve with 30 max iterations and 1e-8 tolerance
10. Build the solver and set verbosity level for output.
11. Print matrix information and start timing measurement.
12. Solve the linear system $Ax = b$.
13. Stop timing measurement and report execution time.
14. Report preconditioner convergence statistics and history.
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

- **Iterative ILU0 Preconditioner**:
  - `rocalution::ItILU0::SetTolerance()`: Sets the convergence tolerance for the iterative ILU0 algorithm.
  - `rocalution::ItILU0::SetMaxIter()`: Sets the maximum number of iterations for the iterative ILU0 algorithm.
  - `rocalution::ItILU0::SetOptions()`: Configures various options including verbosity, stopping criteria, norm computation, and convergence history.
  - `rocalution::ItILU0::SetAlgorithm()`: Sets the algorithm type for iterative triangular solve (SyncSplit).
  - `rocalution::ItILU0::SetSolverDescriptor()`: Configures the solver descriptor for triangular solves.
  - `rocalution::ItILU0::GetConvergenceHistory()`: Retrieves the convergence history of the iterative ILU0 preconditioner.

- **Iterative Triangular Solve**:
  - **Triangular Systems**: Solves systems of the form $Lx = b$ or $Ux = b$ where $L$ and $U$ are lower/upper triangular.
  - **Iterative Approach**: Uses iterative methods instead of direct triangular solves for better parallel performance.
  - **Convergence Monitoring**: Tracks residual reduction and correction norms during iteration.
  - **Synchronization**: SyncSplit algorithm provides synchronization for parallel execution.

- **Solver Descriptor**:
  - `rocalution::SolverDescr::SetTriSolverAlg()`: Sets the triangular solver algorithm to iterative.
  - `rocalution::SolverDescr::SetIterativeSolverMaxIteration()`: Sets maximum iterations for triangular solve.
  - `rocalution::SolverDescr::SetIterativeSolverTolerance()`: Sets tolerance for triangular solve convergence.

- **Fixed-Point Iteration**:
  - `rocalution::FixedPoint::SetRelaxation()`: Sets the relaxation parameter for fixed-point iteration.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
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

- `rocalution::FixedPoint<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::ItILU0<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::SolverDescr`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::FixedPoint::SetRelaxation`
- `rocalution::ItILU0::SetTolerance`
- `rocalution::ItILU0::SetMaxIter`
- `rocalution::ItILU0::SetOptions`
- `rocalution::ItILU0::SetAlgorithm`
- `rocalution::ItILU0::SetSolverDescriptor`
- `rocalution::ItILU0::GetConvergenceHistory`
- `rocalution::SolverDescr::SetTriSolverAlg`
- `rocalution::SolverDescr::SetIterativeSolverMaxIteration`
- `rocalution::SolverDescr::SetIterativeSolverTolerance`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::ItILU0Option`
- `rocalution::ItILU0Algorithm`
- `rocalution::TriSolverAlg`
