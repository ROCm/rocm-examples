# HIP-Doc Call Stack Management Example

## Description

The call stack is a data structure for managing function calls, by saving the state of the current function. Each time
a function is called, a new call frame is added to the top of the stack, containing information such as local
variables, return addresses and function parameters. When the function execution completes, the frame is removed from
the stack and loaded back into the corresponding registers. This concept allows the program to return to the calling
function and continue execution from where it left off.

The call stack for each thread must track its function calls, local variables, and return addresses. However, in GPU
programming, the memory required to store the call stack increases due to the parallelism inherent to the GPUs. NVIDIA
and AMD GPUs use different approaches. NVIDIA GPUs have the independent thread scheduling feature where each thread has
its own call stack and effective program counter. On AMD GPUs threads are grouped; each warp has its own call stack and
program counter.

This example demonstrates how to adjust the call stack size, allowing fine-tuning based on specific kernel
requirements. This helps prevent stack overflow errors by ensuring sufficient stack memory is allocated.

### Application flow

1. The current stack size limit is queried and printed.
2. A new stack size limit is set.
3. The updated stack size limit is queried and printed.

## Key APIs and concepts

* `hipDeviceGetLimit` queries the device for a requested limit, in this case the stack size.
* `hipDeviceSetLimit` sets a new limit for the device.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetLimit`
* `hipDeviceSetLimit`
* `hipGetErrorString`
