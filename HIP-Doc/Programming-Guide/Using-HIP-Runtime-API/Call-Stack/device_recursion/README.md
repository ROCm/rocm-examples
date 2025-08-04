# HIP-Doc Device Recursion Example

## Description

This example demonstrates how to hit the device's stack limit on purpose. For more information on this topic, please
refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/call_stack.html#handling-recursion-and-deep-function-calls).

### Prerequisites

To trigger a stack overflow, uncomment the marked section in `main.hip` and compile without optimizations:

```bash
> hipcc -O0 main.hip # Add other compiler or linker flags as needed
```

### Application flow

1. A GPU kernel is launched which recursively computes a Fibonacci sequence.
2. Device and host are synchronized.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `printf`
* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipGetErrorString`
