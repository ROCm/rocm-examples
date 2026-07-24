# HIP-Basic Execution Context Example

## Description

By default, kernels compete for all of a GPU's compute units (CUs), so a short kernel can be delayed behind a large one that already occupies the device. An execution context binds work to a fixed set of CUs: any kernel on a stream belonging to the context is confined to those CUs, and no kernel source changes are needed. This is HIP's counterpart to CUDA green contexts.

This example measures the same latency-sensitive workload two ways to show the difference. In both cases a long-running background kernel is launched to occupy the device, and a shorter critical kernel is timed while the background kernel runs:

1. **Baseline (shared CUs)**: the two kernels run on ordinary streams and compete for all of the device's CUs, so the critical kernel waits behind the background kernel.
2. **Partitioned (reserved CUs)**: the CUs are split into two execution contexts, the background kernel is confined to the larger group, and the critical kernel runs on its own reserved group, so it is no longer blocked.

The program prints the critical kernel's latency in each case and the resulting speedup.

Execution context resource partitioning is an AMD (HIP) feature. On the CUDA backend, where the required runtime support may be unavailable, only the shared-CU baseline runs; the partitioned measurement is guarded by `__HIP_PLATFORM_AMD__`.

### Application flow

1. The device is selected with `hipSetDevice`.
2. The number of compute units is determined. On the HIP (AMD) backend it comes from `hipDeviceGetDevResource` using the `hipDevResourceTypeSm` resource type. (The field is named `smCount` for CUDA source compatibility; on AMD GPUs it represents compute units.) On other backends it is read from `hipGetDeviceProperties`.
3. **Baseline.** Two ordinary non-blocking streams are created with `hipStreamCreateWithFlags`. The background kernel is launched on one, and the critical kernel is launched and timed on the other with HIP events while the background kernel runs. Both share all CUs.
4. **Partitioned (HIP backend only).** The CU resources are split into two groups with `hipDevSmResourceSplit`: a larger group for the background kernel and a smaller group reserved for the critical kernel.
5. A resource descriptor is generated for each group with `hipDevResourceGenerateDesc`.
6. An execution context is created from each descriptor with `hipGreenCtxCreate`.
7. A stream is created for each execution context with `hipExecutionCtxStreamCreate`, and the same background-plus-critical timing is repeated. The critical kernel now runs on its reserved CUs.
8. The two critical-kernel latencies and the speedup are printed.
9. The device output buffers are freed with `hipFree`, the streams are destroyed with `hipStreamDestroy`, and on the HIP backend the execution contexts are destroyed with `hipExecutionCtxDestroy`.

## Key APIs and Concepts

Execution contexts carve a GPU's CUs into separate slices within one process, so urgent work has resources ready instead of waiting for a busy device to free up. Setting one up is a four-step sequence: read the device resources, split the CU resource, wrap the pieces in a descriptor, and create the context from it. A stream created on the context keeps every kernel launched on it inside that context's CUs. Running the same concurrent background-plus-critical workload with and without partitioning shows the critical kernel's latency drop when it has reserved CUs.

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
- `hipStreamSynchronize`
- `hipStreamDestroy`
- `hipMalloc`
- `hipFree`
- `hipEventCreate`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipEventElapsedTime`
- `hipEventDestroy`
- `hipGetLastError`
