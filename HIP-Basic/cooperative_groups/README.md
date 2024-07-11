# HIP-Basic Cooperative Groups Example

## Description

This program showcases the usage of Cooperative Groups inside a reduction kernel.

Cooperative groups can be used to gain more control over synchronization.

For more insights, you can read the following blog post:
[Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)

### Application flow

1. A number of variables are defined to control the problem details and the kernel launch parameters.
2. Input vector is set up in host memory.
3. The input is copied to the device.
4. The GPU reduction kernel is launched with previously defined arguments.
5. The kernel will perform two reductions: a reduction of the whole threadblock and a reduction of custom partitions.
6. The result vectors are copied back to the host and all device memory is freed.
7. The elements of the result vectors are compared with the expected result. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

Usually, programmers can only synchronize on warp-level or block-level.
But cooperative groups allows the programmer to partition threads together and subsequently synchronize those groups.
The partitioned threads can reside across multiple devices.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `thread_group`
- `thread_block`
- `tiled_partition<size>()`
- `thread_block_tile`
- All above from the [`cooperative_groups` namespace](https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_cooperative_groups.h)

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipLaunchCooperativeKernel`
- `hipDeviceAttributeCooperativeLaunch`
- `hipDeviceGetAttribute`
- `hipGetLastError`
- `hipFree`
