# HIP-Programming-Guide Device Recursion Example

## Description

Similar to CPU programming, recursive functions and deeply nested function calls are supported. However, developers
must ensure that these functions do not exceed the available stack memory, considering the huge amount of memory needed
for the call stack due to the GPUs inherent parallelism. This can be achieved by increasing stack size or optimizing
code to reduce stack usage. To detect stack overflow add proper error handling or use debugging tools.

This example demonstrates how to hit the stack limit.

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
