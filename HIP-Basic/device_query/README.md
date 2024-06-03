# HIP-Basic Device Query Example

## Description

This example shows how the target platform and compiler can be identified, as well as how properties from the device may be queried.

### Application flow

1. Using compiler-defined macros, the target platform and compiler are identified.
2. The number of devices in the system is queried, and for each device:
    1. The device is set as the active device.
    2. The device properties are queried and a selected set is printed.
    3. For each device in the system, it is queried and printed whether this device can access its memory.
    4. If NVIDIA is the target platform, some NVIDIA-specific device properties are printed.
    5. The amount of total and free memory of the device is queried and printed.

## Key APIs and Concepts

- HIP code can target the AMD and the NVIDIA platform, and it can be compiled with different compilers. Compiler-defined macros can be used in HIP code to write code that is specific to a target or a compiler. See [HIP Programming Guide - Distinguishing Compiler Modes](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#distinguishing-compiler-modes) for more details.

- `hipGetDeviceCount` returns the number of devices in the system. Some device management API functions take an identifier for each device, which is a monotonically incrementing number starting from zero. Others require the active device to be set, with `hipSetDevice`. A full overview of the device management API can be found at [HIP API - Device Management](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___device.html).

## Demonstrated API Calls

### HIP Runtime

- `__HIP_PLATFORM_AMD__`
- `__HIP_PLATFORM_NVIDIA__`
- `__CUDACC__`
- `hipGetDeviceCount`
- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipDeviceCanAccessPeer`
- `hipMemGetInfo`
- `cudaDeviceGetLimit`
- `cudaLimit::cudaLimitStackSize`
- `cudaLimit::cudaLimitMallocHeapSize`
