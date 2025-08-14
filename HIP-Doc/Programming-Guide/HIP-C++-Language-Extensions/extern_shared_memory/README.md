# HIP-Doc Extern Shared Memory Example

## Description

This example demonstrates how to dynamically allocate memory in the device's shared memory space. For more information
on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#shared).

### Application flow

1. A device array in the shared memory is declared as `extern`.
2. On the host side, the amount of bytes required for the shared memory array are calculated.
3. Using the triple chevron syntax `<<<>>>` the kernel is started on the device. The number of bytes for the shared
   memory allocation are passed as an additional launch parameter.
4. The host and the device are synchronized.

## Key APIs and Concepts

Use the `extern __shared__` notation to declare a variable or array in the device's shared memory space. The required
amount of bytes must be passed as an additional launch parameter.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipDeviceSynchronize`
