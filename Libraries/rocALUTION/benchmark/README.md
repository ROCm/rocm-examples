# rocALUTION Performance Benchmarks

## Description

This example demonstrates the use of the `rocALUTION` library for performance benchmarking of fundamental linear algebra operations.

The example performs comprehensive micro-benchmarks and combined benchmarks to measure:

1. **Vector Operations**: Dot product, reduce, norm, and vector updates
2. **Sparse Matrix-Vector Multiplication (SpMV)**: Performance across different matrix formats
3. **Memory Bandwidth and Compute Throughput**: GB/s and GFlop/s measurements

This benchmark helps evaluate the performance characteristics of different matrix storage formats and vector operations on the target hardware platform.

## Application flow

1. Parse command line arguments for matrix file path and OMP thread count.
2. Initialize the rocALUTION platform and configure thread settings.
3. Print rocALUTION platform information.
4. Create rocALUTION objects and read the matrix from MTX format file.
5. Allocate vectors and initialize them with test data.
6. Move all objects to the accelerator and print object information.
7. **Stand-alone Micro-benchmarks**:
   - Dot product: 200 iterations measuring memory bandwidth and compute throughput
   - Reduce: 200 iterations measuring reduction operation performance
   - Norm: 200 iterations measuring L2 norm computation performance
   - Vector Update (ScaleAdd): 200 iterations measuring vector arithmetic performance
   - Vector Update (AddScale): 200 iterations measuring alternative vector update performance
8. **Matrix Format SpMV Benchmarks**:
   - Convert matrix to CSR format and benchmark SpMV performance
   - Convert matrix to MCSR format and benchmark SpMV performance
   - Convert matrix to ELL format and benchmark SpMV performance
   - Convert matrix to COO format and benchmark SpMV performance
   - Convert matrix to HYB format and benchmark SpMV performance
   - Convert matrix to DIA format and benchmark SpMV performance
9. **Combined Benchmarks**: 200 iterations of all operations combined to measure overall system performance
10. Report detailed performance metrics including execution time, memory bandwidth, and compute throughput for each operation.
11. Stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`, and OMP thread count can be configured with `rocalution::set_omp_threads_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::ConvertToCSR()`: Converts matrix to Compressed Sparse Row format.
  - `rocalution::LocalMatrix::ConvertToMCSR()`: Converts matrix to Modified Compressed Sparse Row format.
  - `rocalution::LocalMatrix::ConvertToELL()`: Converts matrix to ELL (Ellpack-Itpack) format.
  - `rocalution::LocalMatrix::ConvertToCOO()`: Converts matrix to Coordinate format.
  - `rocalution::LocalMatrix::ConvertToHYB()`: Converts matrix to Hybrid format.
  - `rocalution::LocalMatrix::ConvertToDIA()`: Converts matrix to Diagonal format.
  - `rocalution::LocalMatrix::Apply()`: Performs matrix-vector multiplication $y = Ax$.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including format, dimensions, and non-zero count.
  - `rocalution::LocalMatrix::GetN()`, `rocalution::LocalMatrix::GetM()`, `rocalution::LocalMatrix::GetNnz()`: Return matrix dimensions and non-zero count.

- **Vector Operations**:
  - `rocalution::LocalVector::Allocate()`: Allocates memory for a vector with specified size and name.
  - `rocalution::LocalVector::Ones()`: Initializes all vector elements to value 1.
  - `rocalution::LocalVector::Zeros()`: Initializes all vector elements to value 0.
  - `rocalution::LocalVector::Dot()`: Computes the dot product of two vectors.
  - `rocalution::LocalVector::Reduce()`: Performs reduction operation on vector elements.
  - `rocalution::LocalVector::Norm()`: Computes the L2 norm (Euclidean norm) of the vector.
  - `rocalution::LocalVector::ScaleAdd()`: Computes vector operation $y = \alpha \cdot x + y$.
  - `rocalution::LocalVector::AddScale()`: Computes vector operation $y = x + \alpha \cdot z$.
  - `rocalution::LocalVector::Info()`: Prints vector information including size and location.
  - `rocalution::LocalVector::MoveToAccelerator()`: Transfers vector data from host to device memory.

- **Performance Measurement**:
  - `rocalution::rocalution_time()`: Returns high-resolution timer value for performance measurement.
  - `rocalution::_rocalution_sync()`: Synchronizes device operations to ensure accurate timing.

- **Matrix Storage Formats**:
  - **CSR**: Compressed Sparse Row - efficient for row-wise access patterns
  - **MCSR**: Modified CSR - optimized for certain SpMV patterns
  - **ELL**: Ellpack-Itpack - regular structure optimized for GPU
  - **COO**: Coordinate - simple format for irregular matrices
  - **HYB**: Hybrid - combines ELL and COO for optimal performance
  - **DIA**: Diagonal - efficient for diagonal-dominant matrices

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`
- `rocalution::info_rocalution`
- `rocalution::set_omp_threads_rocalution`
- `rocalution::rocalution_time`
- `rocalution::_rocalution_sync`

### Matrix Operations

- `rocalution::LocalMatrix::ReadFileMTX`
- `rocalution::LocalMatrix::ConvertToCSR`
- `rocalution::LocalMatrix::ConvertToMCSR`
- `rocalution::LocalMatrix::ConvertToELL`
- `rocalution::LocalMatrix::ConvertToCOO`
- `rocalution::LocalMatrix::ConvertToHYB`
- `rocalution::LocalMatrix::ConvertToDIA`
- `rocalution::LocalMatrix::Apply`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::GetN`
- `rocalution::LocalMatrix::GetM`
- `rocalution::LocalMatrix::GetNnz`
- `rocalution::LocalMatrix::MoveToAccelerator`

### Vector Operations

- `rocalution::LocalVector::Allocate`
- `rocalution::LocalVector::Ones`
- `rocalution::LocalVector::Zeros`
- `rocalution::LocalVector::Dot`
- `rocalution::LocalVector::Reduce`
- `rocalution::LocalVector::Norm`
- `rocalution::LocalVector::ScaleAdd`
- `rocalution::LocalVector::AddScale`
- `rocalution::LocalVector::Info`
- `rocalution::LocalVector::MoveToAccelerator`

### Data Types

- `rocalution::LocalMatrix<double>`
- `rocalution::LocalVector<double>`
