# hipBLASLt Extension Operation - Absolute Maximum

## Description

This example illustrates the use of the `hipBLASLt` extension operation for finding the absolute maximum value in a matrix.

The operation calculates the following:

$\text{out} = \max(|A_{i,j}|)$

where

- $A$ is an input matrix of dimensions $m \times n$
- $\text{out}$ is a scalar containing the absolute maximum value of all elements in matrix $A$
- $|A_{i,j}|$ represents the absolute value of element at position $(i,j)$

## Application flow

1. Set up input matrix dimensions and allocate memory for input and output data.
2. Initialize input matrix with random values using the `opt_amax_runner` utility class.
3. Copy input matrix from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Perform the absolute maximum operation using `hipblasltExtAMax()`.
6. Copy the result scalar from device to host memory.
7. Clean up device allocations and destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Execution**:
  - `hipblasltExtAMax()`: Computes the absolute maximum value of a matrix.

- **Key Enumerations**:
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_16F` for half-precision, `HIP_R_32F` for single-precision).

## Demonstrated API Calls

### hipBLASLt

- `hipblasltExtAMax`
- `hipblasLtCreate`
- `hipblasLtDestroy`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types

- `hipDataType`
- `HIP_R_32F`
- `HIP_R_16F`
- `hipblasLtHalf`
- `hipStream_t`
- `hipblasLtHandle_t`
