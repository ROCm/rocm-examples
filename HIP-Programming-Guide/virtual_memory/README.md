# HIP-Programming-Guide Virtual Memory Example

## Description

Memory management is important when creating high-performance applications in the HIP ecosystem. Both allocating and
copying memory can result in bottlenecks, which can significantly impact performance.

Global memory allocation in HIP uses the C language style allocation function. This works fine for simple cases but can
cause problems if your memory needs change. If you need to increase the size of your memory, you must allocate a second
larger buffer and copy the data to it before you can free the original buffer. This increases overall memory usage and
causes unnecessary `memcpy` calls. Another solution is to allocate a larger buffer than you initially need. However,
this isn’t an efficient way to handle resources and doesn’t solve the issue of reallocation when the extra buffer runs
out.

Virtual memory management solves these memory management problems. It helps to reduce memory usage and unnecessary
`memcpy` calls. This example demonstrates how to use HIP's virtual memory management API.

### Prerequisites

The virtual memory management API is currently under development for Windows. For the time being, this example only
works on Linux.

### Application flow

1. Virtual memory support is queried for the current device.
2. Physical memory is allocated using the virtual memory management API.
3. A virtual memory address range is reserved.
4. The physical memory is mapped onto the virtual memory address range.
5. The memory access permissions are set.
6. A memory operation (`memcpy`) is performed.
7. The example kernels are launched and their results are verified.
8. The allocated memory is freed using the virtual memory management API.

## Key APIs and Concepts

* `hipDeviceGetAttribute` is used to query the current device for virtual memory support.
* `hipMemGetAllocationGranularity` is used to obtain the appropriate granularity for memory alignment.
* `hipMemCreate` is used to allocate the physical memory.
* `hipMemAddressReserve` is used to reserve a virtual address range.
* `hipMemMap` maps the physical memory onto the virtual address range.
* `hipMemSetAccess` sets the memory access permissions for the pointer.
* `hipMemcpy` transfers bytes between the host and the device.
* `hipDeviceSynchronize` synchronized the host and the device.
* `hipMemUnmap` unmaps the physical memory.
* `hipMemRelease` frees the physical memory.
* `hipMemAddressFree` frees the virtual address range.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetAttribute`
* `hipDeviceSynchronize`
* `hipGetErrorString`
* `hipMemAddressFree`
* `hipMemAddressReserve`
* `hipMemCreate`
* `hipMemGetAllocationGranularity`
* `hipMemMap`
* `hipMemRelease`
* `hipMemSetAccess`
* `hipMemUnmap`
* `hipMemcpy`
