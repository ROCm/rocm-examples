# HIP-Documentation Dynamic Shared Memory Example

## Description

Shared memory is read-write memory, that is only visible to the threads within a block. It is allocated per thread
block, and needs to be either statically allocated at compile time, or can be dynamically allocated when launching the
kernel, but not during kernel execution. Shared memory can be dynamically allocated by declaring an `extern __shared__`
array, whose size can be set during kernel launch, which can then be accessed in the kernel. This example demonstrates
how to declare such an array and how to allocate the required memory on the host.

### Application flow

1. The required memory size is calculated on the host.
2. The kernel is launched with an additional launch parameter: the bytes of shared memory that are required for the
   kernel.
3. The host checks for any errors.

## Key APIs and Concepts

* `extern __shared__` is used to declare a dynamically allocated shared memory symbol on the device side.
* The third launch parameter of the kernel launch contains the size (in bytes) of the shared memory to allocate.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `threadIdx`

#### Host symbols

* `hipDeviceSynchronize`
* `hipPeekAtLastError`
