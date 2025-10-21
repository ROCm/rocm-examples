# rocALUTION Cuthill-McKee Ordering

## Description

This example demonstrates the use of the `rocALUTION` library for solving symmetric positive definite linear systems using the Conjugate Gradient method with Cuthill-McKee (CMK) ordering optimization.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a symmetric positive definite matrix
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how to apply Reverse Cuthill-McKee (RCMK) ordering to improve matrix structure before solving. The CMK ordering reduces the bandwidth of the matrix, which can significantly improve the performance of direct and iterative solvers, particularly those with ILU preconditioning. After solving, the solution vector is permuted back to the original ordering.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Compute Reverse Cuthill-McKee ordering for the matrix.
6. Move all objects to the accelerator (GPU).
7. Apply the RCMK ordering to the matrix to improve its structure.
8. Allocate solution, RHS, and error vectors with appropriate dimensions.
9. Initialize the right-hand side vector such that $A \cdot 1 = b$.
10. Set initial solution guess to zero vector.
11. Configure the CG solver with ILU preconditioner.
12. Build the solver and set verbosity level for output.
13. Print matrix information and start timing measurement.
14. Solve the linear system $Ax = b$ with the reordered matrix.
15. Stop timing measurement and report execution time.
16. Revert the RCMK ordering on the solution vector to restore original indexing.
17. Compute and report the L2 norm of the error.
18. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::RCMK()`: Computes Reverse Cuthill-McKee ordering for the matrix.
  - `rocalution::LocalMatrix::Permute()`: Applies a permutation to the matrix rows and columns.
  - `rocalution::LocalMatrix::MoveToAccelerator()`: Transfers matrix data from host to device memory.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and non-zero count.
  - `rocalution::LocalMatrix::GetN()` and `rocalution::LocalMatrix::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector<int>::Allocate()`: Allocates memory for an integer vector for ordering.
  - `rocalution::LocalVector<double>::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector<double>::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector<double>::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector<double>::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector<double>::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.
  - `rocalution::LocalVector<double>::PermuteBackward()`: Applies inverse permutation to restore original ordering.
  - `rocalution::LocalVector::MoveToAccelerator()`: Transfers vector data from host to device memory.

- **Cuthill-McKee Ordering**:
  - **Bandwidth Reduction**: Reorders matrix to minimize bandwidth and profile.
  - **Reverse CMK**: Uses reverse ordering algorithm for better cache performance.
  - **Fill Reduction**: Reduces the number of non-zero elements within the matrix profile.
  - **Solver Performance**: Improved matrix structure leads to better solver convergence and memory access patterns.

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
- `rocalution::LocalMatrix::RCMK`
- `rocalution::LocalMatrix::Permute`
- `rocalution::LocalMatrix::MoveToAccelerator`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::GetM`

### Vector Operations

- `rocalution::LocalVector<int>::Allocate`
- `rocalution::LocalVector<double>::Allocate`
- `rocalution::LocalVector<double>::Ones`
- `rocalution::LocalVector<double>::Zeros`
- `rocalution::LocalVector<double>::ScaleAdd`
- `rocalution::LocalVector<double>::Norm`
- `rocalution::LocalVector<double>::PermuteBackward`
- `rocalution::LocalVector<double>::MoveToAccelerator`

### Solver Classes

- `rocalution::CG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`
- `rocalution::ILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
- `rocalution::LocalVector<int>`
