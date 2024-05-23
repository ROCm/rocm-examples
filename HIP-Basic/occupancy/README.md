# HIP-Basic Occupancy Example

## Description

This example showcases how to find optimal configuration parameters for a kernel launch with maximum occupancy. It uses the HIP occupancy calculator APIs to find a kernel launch configuration that yields maximum occupancy. This configuration is used to launch a kernel and measures the utilization difference against another kernel launch that is manually (and suboptimally) configured. The application kernel is a simple vector--vector multiplication of the form `C[i] = A[i]*B[i]`, where `A`, `B` and `C` are vectors of size `size`.

The example shows 100% occupancy for both manual and automatic configurations, because the simple kernel does not use much resources per-thread or per-block, especially `__shared__` memory. The execution time for the automatic launch is still lower because of a lower overhead associated with fewer blocks being executed.

### Application flow

1. Host side data is instantiated in `std::vector<float>`.
2. Device side storage is allocated using `hipMalloc` in `float*`.
3. Data is copied from host to device using `hipMemcpy`.
4. Kernel is launched using a manual default block size of `32`. Maximum occupany is found using `hipOccupancyMaxActiveBlocksPerMultiprocessor` against the current block size. It is reported as percentage of the theoretical maximum possible.
5. The time spent for kernel execution is aquired using `hipEventElapsedTime`. It is then printed on the screen.
6. The same kernel is launched again, but this time with the block size that is found using `hipOccupancyMaxPotentialBlockSize`. Again, maximum occupany is found using `hipOccupancyMaxActiveBlocksPerMultiprocessor` against the current block size. It is reported as percentage of the theoretical maximum possible.
7. The time spent for kernel execution is aquired using `hipEventElapsedTime`. It is then printed on the screen.
8. Result is transferred from device to host.
9. All device memory is freed using `hipFree`.

## Key APIs and Concepts

GPUs have large amount of parallel resources available. Utilizing these resources in an optimal way is very important to achieve best performance. The HIP occupancy calculator API `hipOccupancyMaxPotentialBlockSize` allows finding kernel block size that launches most amount of threads per thread block for a given kernel. The `hipOccupancyMaxActiveBlocksPerMultiprocessor` calculates maximum active blocks per GPU multiprocessor for a given block size and kernel.

### Occupancy

Occupancy is the ratio of active wavefronts (or warps) to the maximum number of wavefronts (or warps) that can be deployed on a GPU multiprocessor. HIP GPU threads execute on a GPU multiprocessor, which has limited resources such as registers and shared memory. These resources are shared among threads within a thread block. When the usage of these shared resources is minimized (by compiler optimization or user code design) more blocks can simultaneously execute per multiprocessor thereby increasing the occupancy.

## Used API surface

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipEventCreate`
- `hipOccupancyMaxPotentialBlockSize`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`
- `hipGetDeviceProperties`
- `hipOccupancyMaxActiveBlocksPerMultiprocessor`
- `hipFree`
