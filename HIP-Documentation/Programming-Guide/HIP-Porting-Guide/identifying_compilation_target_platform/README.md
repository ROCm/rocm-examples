# HIP-Documentation Identifying Compilation Target Platform Example

## Description

This example demonstrates how to use preprocessor macros to distinguish between AMD and NVIDIA compilation targets. For
more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#identifying-the-hip-target-platform).

### Application flow

There is no application flow in this example.

## Key APIs and Concepts

* `__HIP_PLATFORM_AMD__` is defined whenever ROCm's clang-based compiler parses a code file.
* `__HIP_PLATFORM_NVIDIA__` is defined whenever CUDA's `nvcc` compiler parses a code file.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `__HIP_PLATFORM_AMD__`
* `__HIP_PLATFORM_NVIDIA__`
