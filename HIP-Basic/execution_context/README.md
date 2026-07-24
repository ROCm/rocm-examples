# HIP-Basic Execution Context Example

## Description

By default, kernels compete for all of a GPU's compute units (CUs), so a short kernel can be delayed behind a large one that already occupies the device. An execution context binds work to a fixed set of CUs: any kernel on a stream belonging to the context is confined to those CUs, and no kernel source changes are needed. This example splits a device's CUs into two contexts, gives most of them to a long-running kernel and reserves a small set for a shorter critical kernel, and times both to show that the critical kernel is no longer blocked. This is HIP's counterpart to CUDA green contexts.

Execution context resource partitioning is an AMD (HIP) feature. On the CUDA backend, where the required runtime support may be unavailable, the example compiles and runs using two ordinary non-blocking streams without CU partitioning, guarded by `__HIP_PLATFORM_AMD__`.

### Application flow

1. The device is selected with `hipSetDevice`.
2. The number of compute units is determined. On the HIP (AMD) backend it comes from `hipDeviceGetDevResource` using the `hipDevResourceTypeSm` resource type. (The field is named `smCount` for CUDA source compatibility; on AMD GPUs it represents compute units.) On other backends it is read from `hipGetDeviceProperties`.
3. On the HIP backend, the CU resources are split into two groups with `hipDevSmResourceSplit`: a larger group for the long-running kernel and a smaller group reserved for the critical kernel.
4. A resource descriptor is generated for each group with `hipDevResourceGenerateDesc`.
5. An execution context is created from each descriptor with `hipGreenCtxCreate`.
6. A stream is created for each execution context with `hipExecutionCtxStreamCreate`. On non-HIP backends, two ordinary non-blocking streams are created with `hipStreamCreateWithFlags` instead, and steps 3 to 5 are skipped.
7. A busy kernel is launched and timed on each stream using HIP events. The long-running kernel oversubscribes the CUs; the critical kernel runs a shorter workload.
8. The device output buffers are freed with `hipFree`.
9. The streams are destroyed with `hipStreamDestroy`.
10. On the HIP backend, the execution contexts are destroyed with `hipExecutionCtxDestroy`.

## Key APIs and Concepts

Execution contexts carve a GPU's CUs into separate slices within one process, so urgent work has resources ready instead of waiting for a busy device to free up. Setting one up is a four-step sequence: read the device resources, split the CU resource, wrap the pieces in a descriptor, and create the context from it. A stream created on the context keeps every kernel launched on it inside that context's CUs.

## Demonstrated API Calls

### HIP runtime

- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipDeviceGetDevResource`
- `hipDevResourceTypeSm`
- `hipDevResource`
- `hipDevSmResourceGroupParams`
- `hipDevSmResourceSplit`
- `hipDevResourceGenerateDesc`
- `hipDevResourceDesc_t`
- `hipGreenCtxCreate`
- `hipExecutionCtx_t`
- `hipExecutionCtxStreamCreate`
- `hipExecutionCtxDestroy`
- `hipStream_t`
- `hipStreamCreateWithFlags`
- `hipStreamDestroy`
- `hipMalloc`
- `hipFree`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`
- `hipGetLastError`
