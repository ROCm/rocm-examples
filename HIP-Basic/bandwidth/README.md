# Cookbook Bandwidth Example

## Description

This example measures the memory bandwith capacity of GPU devices. It performs memcpy from host to GPU device, GPU device to host, and within a single GPU.

### Application flow

1. User commandline arguments are parsed and test parameters initialized. If there are no commandline arguments then the test paramenters are initialized with default values.
2. Bandwidth tests are launched.
3. If the memory type for the test set to `-memory pageable` then the host side data is instantiated in `std::vector<unsigned char>`. If the memory type for the test set to `-memory pinned` then the host side data is instantiated in `unsigned char*` and allocated using `hipHostMalloc`.
4. Device side storage is allocated using `hipMalloc` in `unsigned char*`
5. Memory transfer is performed `trail` amount of times using `hipMemcpy` for pageable memory or using `hipMemcpyAsync` for host allocated pinned memory.
6. Time of memory transfer operations is measured that is then used to calculate the bandwidth.
7. All device memory is freed using `hipFree` and all host allocated pinned memory is freed using `hipHostFree`.

## Key APIs and Concepts

The program uses HIP pageable and pinned memory. It is important to note that the pinned memory is allocated using `hipHostMalloc` and is destroyed using `hipHostFree`. The HIP memory transfer routine `hipMemcpyAsync` will behave synchronously if the host memory is not pinned. Therefore, it is important to allocate pinned host memory using `hipHostMalloc` for `hipMemcpyAsync` to behave asynchronously.

## Demonstrated API Calls

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyAsync`
- `hipGetDeviceCount`
- `hipGetDeviceProperties`
- `hipFree`
- `hipHostFree`
- `hipHostMalloc`
- `hipSetDevice`
