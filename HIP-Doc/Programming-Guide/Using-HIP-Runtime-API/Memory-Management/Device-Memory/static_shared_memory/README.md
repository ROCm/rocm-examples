# HIP-Doc Static Shared Memory Example

## Description

To statically allocate shared memory, just declare it in the kernel. This example demonstrates how this is achieved.

### Application flow

1. The kernel is launched.
2. The host checks for any errors.

## Key APIs and Concepts

* `__shared__` is used to declare a statically allocated shared memory symbol on the device side.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceSynchronize`
* `hipPeekAtLastError`
