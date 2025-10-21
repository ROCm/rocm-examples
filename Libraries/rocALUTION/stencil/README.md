# rocALUTION Stencil-Based Linear Solver

## Description

This example demonstrates the use of the `rocALUTION` library for solving linear systems arising from stencil operations using the Conjugate Gradient method.

The operation solves the linear system:

$Ax = b$

where

- $A$ is a matrix implicitly defined by a stencil operator
- $x$ is the solution vector
- $b$ is the right-hand side vector

This example shows how rocALUTION handles stencil-based operators, which are commonly used in computational fluid dynamics, heat transfer simulations, and image processing. The stencil represents the computational pattern used to update each grid point based on its neighbors.

## Application flow

1. Initialize the rocALUTION platform.
2. Print rocALUTION platform information.
3. Create rocALUTION objects including a 2D Laplace stencil.
4. Set up the stencil grid to 100×100 points.
5. Allocate solution, RHS, and error vectors with appropriate dimensions.
6. Initialize the right-hand side vector by applying the stencil to a vector of ones.
7. Set initial solution guess to zero vector.
8. Configure the CG solver to work with the stencil operator.
9. Build the solver and set verbosity level for output.
10. Print stencil information and start timing measurement.
11. Solve the linear system $Ax = b$ using the stencil operator.
12. Stop timing measurement and report execution time.
13. Compute and report the L2 norm of the error.
14. Clear solver resources and stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`.

- **Stencil Operations**:
  - `rocalution::LocalStencil<double>::LocalStencil()`: Creates a stencil object with specified type (Laplace2D).
  - `rocalution::LocalStencil<double>::SetGrid()`: Sets the grid dimensions for the stencil operation.
  - `rocalution::LocalStencil<double>::Apply()`: Applies the stencil operator to a vector.
  - `rocalution::LocalStencil<double>::Info()`: Prints stencil information including grid dimensions and operator type.
  - `rocalution::LocalStencil<double>::GetN()` and `rocalution::LocalStencil<double>::GetM()`: Return stencil dimensions (columns and rows respectively).

- **Vector Operations**:
  - `rocalution::LocalVector<double>::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector<double>::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector<double>::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector<double>::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector<double>::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.

- **Stencil-Based Solver Configuration**:
  - `rocalution::IterativeLinearSolver::SetOperator()`: Associates the stencil operator with the solver.
  - `rocalution::IterativeLinearSolver::Build()`: Initializes solver data structures for stencil operations.
  - `rocalution::IterativeLinearSolver::Verbose()`: Sets the verbosity level for solver progress output.
  - `rocalution::IterativeLinearSolver::Solve()`: Executes the iterative solver to find the solution.
  - `rocalution::IterativeLinearSolver::Clear()`: Releases solver resources and memory.

- **Stencil Operators**:
  - **Laplace2D**: 2D Laplacian operator representing the finite difference approximation of the Laplace operator $\nabla^2$ on a 2D grid.
  - **Implicit Matrix**: The stencil operator implicitly defines a matrix without explicit storage, improving memory efficiency for structured problems.

- **Performance Measurement**:
  - `rocalution::rocalution_time()`: Returns high-resolution timer value for performance measurement.

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`
- `rocalution::info_rocalution`
- `rocalution::rocalution_time`

### Stencil Operations

- `rocalution::LocalStencil<double>::LocalStencil`
- `rocalution::LocalStencil<double>::SetGrid`
- `rocalution::LocalStencil<double>::Apply`
- `rocalution::LocalStencil<double>::Info`
- `rocalution::LocalStencil<double>::GetN`
- `rocalution::LocalStencil<double>::GetM`

### Vector Operations

- `rocalution::LocalVector<double>::Allocate`
- `rocalution::LocalVector<double>::Ones`
- `rocalution::LocalVector<double>::Zeros`
- `rocalution::LocalVector<double>::ScaleAdd`
- `rocalution::LocalVector<double>::Norm`

### Solver Classes

- `rocalution::CG<rocalution::LocalStencil<double>, rocalution::LocalVector<double>, double>`

### Solver Methods

- `rocalution::IterativeLinearSolver::SetOperator`
- `rocalution::IterativeLinearSolver::Build`
- `rocalution::IterativeLinearSolver::Verbose`
- `rocalution::IterativeLinearSolver::Solve`
- `rocalution::IterativeLinearSolver::Clear`

### Data Types

- `rocalution::LocalStencil<double>`
- `rocalution::LocalVector<double>`
