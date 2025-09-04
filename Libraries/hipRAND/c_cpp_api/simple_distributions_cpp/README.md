# hipRAND c_cpp_api simple_distributions Example

## Description

This example illustrates the use of the hipRAND cpp API. It specifically shows an example for a simple distribution.

### Application flow

1. Parse command-line arguments:
    - Device ID
    - Random distribution type
    - Problem size
    - Print toggle
2. Query and set the selected HIP device.
3. Generate random numbers on the device (GPU) using hipRAND.
4. Generate random numbers on the host (CPU) using the standard library.
5. Measure and print execution time for both device and host.
6. Optionally print the generated random numbers.

## Key APIs and Concepts

### hiprand_cpp

- The host level API of hipRAND is used to generate different random distributions using the GPU (in this case `hiprand_cpp::uniform_int_distribution<unsigned int>`, `hiprand_cpp::uniform_real_distribution<float>`, `hiprand_cpp::normal_distribution<double>` and `hiprand_cpp::poisson_distribution<unsigned int>`).

## Used API surface

### hiprand_cpp

- `default_random_engine`
- `uniform_int_distribution`
- `uniform_real_distribution`
- `normal_distribution`
- `poisson_distribution`

### HIP runtime

- `HIP_CHECK`
- `hipSetDevice`
- `hipDeviceProp_t`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipGetDeviceCount`
