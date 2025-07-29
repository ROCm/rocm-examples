# HIP-Documentation Virtual Memory Example

## Description

This example demonstrates how to use HIP's virtual memory management API. For more information on the
topic of virtual memory management, please refer to the
[documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html).

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
