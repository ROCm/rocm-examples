# HIP-Doc Pointer Memory Type Example

## Description

This example demonstrates how to query a pointer's memory type. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_driver_api.html#cu-pointer-attribute-memory-type).

### Application flow

1. A device buffer is allocated.
2. The device pointer's attributes are queried.
3. The device pointer's memory type attribute is printed.
4. A host buffer is allocated.
5. The host pointer's attributes are queried.
6. The host pointer's memory type attribute is printed.

## Key APIs and Concepts

* `hipPointerGetAttributes` is used to query a given pointer's attributes.
* `hipPointerAttribute_t::type` contains information about a pointer's memory type.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMallocHost`
* `hipPointerGetAttributes`
