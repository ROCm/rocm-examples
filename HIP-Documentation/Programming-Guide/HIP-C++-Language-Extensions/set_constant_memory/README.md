# HIP-Documentation Set Constant Memory Example

## Description

This example demonstrates how to initialize memory in the device's constant memory space. For more information on this
topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#constant).

### Application flow

1. A device array in constant memory is statically allocated.
2. A host array is defined.
3. The host array is copied to the device array.

## Key APIs and Concepts

* Use `hipMemcpyToSymbol` to copy bytes from the host to an address in the device's constant memory space.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipMemcpyToSymbol`
