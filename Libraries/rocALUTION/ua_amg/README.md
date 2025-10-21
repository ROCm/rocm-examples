# rocALUTION Unsmoothed Aggregation Algebraic Multigrid (UAAMG)

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems using Unsmoothed Aggregation Algebraic Multigrid method.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

The Unsmoothed Aggregation Algebraic Multigrid method is a multigrid algorithm that constructs a hierarchy of coarser grids without smoothing the interpolation operators. This approach can be more robust for certain problem types and uses PMIS (Parallel Modified Independent Set) coarsening strategy for optimal parallel performance.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Allocate solution, RHS, and error vectors with appropriate dimensions.
6. Initialize the right-hand side vector such that $A \cdot 1 = b$.
7. Set initial solution guess to zero vector.
8. Configure the UAAMG solver with specific parameters:
   - Set coupling strength to 0.001
   - Set coarsest level to 200 unknowns
   - Enable manual smoothers and solver
   - Enable grid transfer scaling
   - Set coarsening strategy to PMIS (Parallel Modified Independent Set)
9. Move all objects to the accelerator and build the AMG hierarchy.
10. Obtain the number of AMG levels and configure coarse grid solver (CG).
11. Set up smoothers for each level using Fixed-Point iteration with Jacobi preconditioners.
12. Configure smoother pre and post iteration counts.
13. Initialize solver tolerances and build the complete solver.
14. Print matrix information and measure build and solve times.
15. Solve the linear system $Ax = b$.
16. Clear solver resources and free allocated smoother objects.
17. Compute and report the L2 norm of the error.
18. Stop the rocALUTION platform.

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

- **UAAMG Configuration**:
  - `rocalution::UAAMG::SetCouplingStrength()`: Sets the coupling strength for aggregation.
  - `rocalution::UAAMG::SetCoarsestLevel()`: Sets the maximum number of unknowns on the coarsest level.
  - `rocalution::UAAMG::SetManualSmoothers()`: Enables manual configuration of smoothers.
  - `rocalution::UAAMG::SetManualSolver()`: Enables manual configuration of coarse grid solver.
  - `rocalution::UAAMG::SetScaling()`: Enables grid transfer scaling.
  - `rocalution::UAAMG::SetCoarseningStrategy()`: Sets the coarsening strategy (PMIS).
  - `rocalution::UAAMG::BuildHierarchy()`: Constructs the multigrid hierarchy.
  - `rocalution::UAAMG::GetNumLevels()`: Returns the number of AMG levels.
  - `rocalution::UAAMG::SetSmoother()`: Sets the smoothers for each level.
  - `rocalution::UAAMG::SetSolver()`: Sets the coarse grid solver.
  - `rocalution::UAAMG::SetSmootherPreIter()`: Sets the number of pre-smoothing steps.
  - `rocalution::UAAMG::SetSmootherPostIter()`: Sets the number of post-smoothing steps.
  - `rocalution::UAAMG::MoveToAccelerator()`: Moves the AMG solver to accelerator.

- **Unsmoothed Aggregation**:
  - **No Smoothing**: Unlike SAAMG, the interpolation operators are not smoothed, which can improve robustness.
  - **PMIS Strategy**: Parallel Modified Independent Set coarsening for optimal parallel performance.
  - **Memory Efficiency**: Unsmoothed interpolation can reduce memory requirements.

- **Fixed-Point Iteration**:
  - `rocalution::FixedPoint::SetRelaxation()`: Sets the relaxation parameter for fixed-point iteration.

- **Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Init()`: Sets solver tolerances and iteration limits.
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

- `rocalution::UAAMG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::CG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::FixedPoint<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::Jacobi<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Init`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`
- `rocalution::UAAMG::SetCouplingStrength`
- `rocalution::UAAMG::SetCoarsestLevel`
- `rocalution::UAAMG::SetManualSmoothers`
- `rocalution::UAAMG::SetManualSolver`
- `rocalution::UAAMG::SetScaling`
- `rocalution::UAAMG::SetCoarseningStrategy`
- `rocalution::UAAMG::BuildHierarchy`
- `rocalution::UAAMG::GetNumLevels`
- `rocalution::UAAMG::SetSmoother`
- `rocalution::UAAMG::SetSolver`
- `rocalution::UAAMG::SetSmootherPreIter`
- `rocalution::UAAMG::SetSmootherPostIter`
- `rocalution::UAAMG::MoveToAccelerator`
- `rocalution::FixedPoint::SetRelaxation`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::IterativeLinearSolver<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::CoarseningStrategy`
