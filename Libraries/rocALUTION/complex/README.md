# rocALUTION Complex-Valued Linear Solver

## Description

This example demonstrates the use of the `rocALUTION` library for solving complex-valued linear systems using the Induced Dimension Reduction method.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a complex-valued matrix
- $x$ is the complex-valued solution vector
- $b$ is the complex-valued right-hand side vector

This example shows how rocALUTION handles complex-valued matrices and vectors, which are essential for many scientific and engineering applications including signal processing, quantum mechanics, and electromagnetic simulations.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create complex-valued rocALUTION objects (vectors and matrix).
5. Read the complex-valued matrix from MTX format file.
6. Move all objects to the accelerator (GPU).
7. Allocate solution, RHS, and error vectors with appropriate dimensions.
8. Initialize the error vector with complex values (1.0 - 1.0i) and move to accelerator.
9. Initialize the right-hand side vector by applying the matrix to the error vector.
10. Set initial solution guess to zero vector.
11. Configure the IDR solver with complex-valued Jacobi preconditioner.
12. Build the solver and set verbosity level for output.
13. Print matrix information and start timing measurement.
14. Solve the complex-valued linear system $Ax = b$.
15. Stop timing measurement and report execution time.
16. Compute and report the L2 norm of the complex-valued error.
17. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Complex Matrix Operations**:
  - `rocalution::LocalMatrix<std::complex<double>>::ReadFileMTX()`: Loads a complex-valued matrix from Matrix Market format file.
  - `rocalution::LocalMatrix<std::complex<double>>::MoveToAccelerator()`: Transfers complex matrix data from host to device memory.
  - `rocalution::LocalMatrix<std::complex<double>>::Apply()`: Performs complex matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix<std::complex<double>>::Info()`: Prints complex matrix information including dimensions and format.
  - `rocalution::LocalMatrix<std::complex<double>>::GetN()` and `rocalution::LocalMatrix<std::complex<double>>::GetM()`: Return matrix dimensions (columns and rows respectively).

- **Complex Vector Operations**:
  - `rocalution::LocalVector<std::complex<double>>::Allocate()`: Allocates memory for a complex vector with specified size and name.
  - `rocalution::LocalVector<std::complex<double>>::Zeros()`: Initializes all complex vector elements to zero.
  - `rocalution::LocalVector<std::complex<double>>::ScaleAdd()`: Computes complex vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector<std::complex<double>>::Norm()`: Computes the L2 norm of the complex vector.
  - `rocalution::LocalVector<std::complex<double>>::MoveToAccelerator()`: Transfers complex vector data from host to device memory.

- **Complex Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the complex linear system matrix with the solver.
  - `rocalution::IterativeLinearSolver::SetPreconditioner()`: Configures the complex preconditioner for accelerated convergence.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures and complex preconditioner.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the complex solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Complex-Valued Operations**:
  - **Complex Arithmetic**: All operations handle complex numbers with both real and imaginary parts.
  - **Complex Norm**: Computes $\sqrt{\sum |x_i|^2}$ where $x_i$ are complex elements.
  - **Complex SpMV**: Performs $y = Ax$ where $A$, $x$, and $y$ contain complex values.

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

- `rocalution::LocalMatrix<std::complex<double>>::ReadFileMTX`
- `rocalution::LocalMatrix<std::complex<double>>::MoveToAccelerator`
- `rocalution::LocalMatrix<std::complex<double>>::Apply`
- `rocalution::LocalMatrix<std::complex<double>>::Info`
- `rocalution::LocalMatrix<std::complex<double>>::GetN`
- `rocalution::LocalMatrix<std::complex<double>>::GetM`

### Vector Operations

- `rocalution::LocalVector<std::complex<double>>::Allocate`
- `rocalution::LocalVector<std::complex<double>>::Zeros`
- `rocalution::LocalVector<std::complex<double>>::ScaleAdd`
- `rocalution::LocalVector<std::complex<double>>::Norm`
- `rocalution::LocalVector<std::complex<double>>::MoveToAccelerator`

### Solver Classes

- `rocalution::IDR<rocalution::LocalMatrix<std::complex<double>>, rocalution::LocalVector<std::complex<double>>, std::complex<double>>`
- `rocalution::Jacobi<rocalution::LocalMatrix<std::complex<double>>, rocalution::LocalVector<std::complex<double>>, std::complex<double>>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::SetPreconditioner`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`

### Data Types

- `rocalution::LocalMatrix<std::complex<double>>`
- `rocalution::LocalVector<std::complex<double>>`
- `std::complex<double>`
