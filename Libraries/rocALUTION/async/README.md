# rocALUTION Asynchronous Operations

## Description

This example demonstrates the use of the `rocALUTION` library for performing asynchronous memory transfers and computations to overlap data movement with computation.

The example performs matrix-vector operations with different memory transfer strategies:

1. **Synchronous Transfers**: Traditional blocking transfers where computation waits for data movement completion
2. **Asynchronous Transfers**: Non-blocking transfers that allow computation to proceed while data is being transferred
3. **Performance Comparison**: Measures execution time differences between synchronous and asynchronous approaches

This example shows how to optimize performance by overlapping CPU computation with GPU memory transfers, a common technique in high-performance computing.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Allocate vectors and initialize the input vector with ones.
6. **Synchronous Baseline Test**:
   - Perform 100 iterations of matrix-vector multiplication on CPU
   - Move all objects to accelerator synchronously
   - Perform 100 iterations on accelerator
   - Measure total execution time
7. **Asynchronous Test**:
   - Transfer matrix and input vector to accelerator asynchronously
   - Perform CPU computation while data is being transferred
   - Synchronize objects and transfer remaining vector
   - Perform accelerator computation
   - Measure total execution time
8. Compare performance between synchronous and asynchronous approaches.
9. Stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::MoveToAccelerator()`: Synchronously transfers matrix data from host to device memory.
  - `rocalution::LocalMatrix::MoveToAcceleratorAsync()`: Asynchronously transfers matrix data from host to device memory.
  - `rocalution::LocalMatrix::MoveToHost()`: Transfers matrix data from device to host memory.
  - `rocalution::LocalMatrix::ApplyAdd()`: Performs matrix-vector multiplication with accumulation: $y = y + \alpha \cdot Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and format.
  - `rocalution::LocalMatrix::Sync()`: Synchronizes asynchronous memory transfers for the matrix.

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector::MoveToAccelerator()`: Synchronously transfers vector data from host to device memory.
  - `rocalution::LocalVector::MoveToAcceleratorAsync()`: Asynchronously transfers vector data from host to device memory.
  - `rocalution::LocalVector::MoveToHost()`: Transfers vector data from device to host memory.
  - `rocalution::LocalVector::Dot()`: Computes the dot product of two vectors.
  - `rocalution::LocalVector::Info()`: Prints vector information including size and location.
  - `rocalution::LocalVector::Sync()`: Synchronizes asynchronous memory transfers for the vector.

- **Asynchronous Operations**:
  - **Non-blocking Transfers**: `MoveToAcceleratorAsync()` allows computation to continue while data is being transferred.
  - **Synchronization**: `Sync()` ensures that asynchronous transfers are completed before dependent operations.
  - **Overlap**: CPU computation can proceed while GPU memory transfers are in progress.
  - **Performance Optimization**: Asynchronous operations can significantly reduce total execution time by hiding memory transfer latency.

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
- `rocalution::LocalMatrix::MoveToAcceleratorAsync`
- `rocalution::LocalMatrix::MoveToHost`
- `rocalution::LocalMatrix::ApplyAdd`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::Sync`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::Zeros`
- `rocalution::LocalVector::MoveToAccelerator`
- `rocalution::LocalVector::MoveToAcceleratorAsync`
- `rocalution::LocalVector::MoveToHost`
- `rocalution::LocalVector::Dot`
- `rocalution::LocalVector::Info`
- `rocalution::LocalVector::Sync`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
