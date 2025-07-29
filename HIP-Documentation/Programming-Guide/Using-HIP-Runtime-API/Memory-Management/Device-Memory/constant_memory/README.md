# HIP-Documentation Constant Memory Example

## Description

When not using
[unified memory management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html),
memory has to be explicitly copied between the device and the host, using the HIP runtime API. This example demonstrates
how to transfer bytes between the host and the device.

### Application flow

1. A buffer is created on the host.
2. An input and an output buffer are created on the device.
3. The host buffer is copied to the device's input buffer.
4. A placeholder comment simulates a kernel launch on the device.
5. The device's output buffer is copied to the host's buffer.
6. The device memory is freed.
7. The host memory is freed.

## Key APIs and Concepts

* `hipMalloc` is used to allocate memory on the device.
* `hipMemcpy` is used to copy memory from the host to the device and vice versa.
* `hipMemcpyToSymbol` is used to copy data from host memory to a symbol on the device, which can be defined in constant
  or device memory space. The symbol name must be enclosed in the HIP_SYMBOL macro.
* `hipFree` is used to free memory on the device.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `threadIdx`
* `warpSize`

#### Host symbols

* `hipFree`
* `hipMalloc`
* `hipMemcpy`
* `hipMemcpyToSymbol`
