# hipRAND Device API Pseudorandom Generation Example

## Description

This example illustrates the use of the hipRAND device API. It specifically shows an example for a pseudorandom generator inside a kernel.

### Application flow

1. Set pseudorandom generator type (hiprandStateXORWOW).
2. Allocate device memory for the output buffer.
3. Launch the hiprand_kernel on the GPU to generate random numbers.
    - Use `hiprand_init` to initialize the generator.
    - Use `hiprand` to get the next number from the generator
4. Copy generated random numbers from device memory to host memory.
5. Free device memory.
6. Validate the uniformity of the generated random numbers:
    - Compute normalized mean of the distribution.
    - Compare against expected ~0.5.
    - Report validation result (success or failure).

## Key APIs and Concepts

### hipRAND

- The device level API of hipRAND is used to create a generator for pseudorandom numbers. In this case the `hiprandStateXORWOW` generator is used.

## Used API surface

### hipRAND

- `hiprandStateXORWOW`
- `hiprand_init`
- `hiprand`

### HIP runtime

- `HIP_CHECK`
- `hipMalloc`
- `hipDeviceSynchronize`
- `hipGetLastError`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipFree`
