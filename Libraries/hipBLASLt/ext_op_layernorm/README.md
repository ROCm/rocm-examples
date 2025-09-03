# hipBLASLt Extension Operation - Layer Normalization

## Description

This example illustrates the use of the `hipBLASLt` extension operation for layer normalization.

The operation calculates the following:

$\text{out}_{i,j} = \gamma_j \cdot \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j$

where

- $x$ is the input matrix of dimensions $m \times n$
- $\text{out}$ is the output matrix of dimensions $m \times n$
- $\mu_i$ is the mean of the $i$-th row: $\mu_i = \frac{1}{n} \sum_{j=1}^{n} x_{i,j}$
- $\sigma_i^2$ is the variance of the $i$-th row: $\sigma_i^2 = \frac{1}{n} \sum_{j=1}^{n} (x_{i,j} - \mu_i)^2$
- $\gamma$ and $\beta$ are learnable parameters (scale and shift vectors of length $n$)
- $\epsilon$ is a small constant for numerical stability

## Application flow

1. Set up input matrix dimensions and allocate memory for input, output, and parameter data.
2. Initialize input matrix and learnable parameters with random values using the `layer_norm_runner` utility class.
3. Copy input matrix and parameters from host to device memory.
4. Set up hipBLASLt handle and stream.
5. Perform the layer normalization operation using `hipblasltExtLayerNorm()`.
6. Copy the result matrices (output, mean, and inverse variance) from device to host memory.
7. Clean up device allocations and destroy hipBLASLt handle.

## Key APIs and Concepts

- **hipBLASLt Initialization**: The hipBLASLt library is initialized by creating a handle with `hipblasLtCreate()` and released with `hipblasLtDestroy()`.

- **Execution**:
  - `hipblasltExtLayerNorm()`: Performs layer normalization on a matrix.

- **Key Enumerations**:
  - `hipDataType`: Defines the data type of matrices (e.g., `HIP_R_32F` for single-precision).

## Demonstrated API Calls

### hipBLASLt

- `hipblasltExtLayerNorm`
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
- `hipStream_t`
- `hipblasLtHandle_t`
