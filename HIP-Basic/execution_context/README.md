# HIP-Basic Execution Context Example

## Description

By default, kernels compete for all of a GPU's compute units (CUs), so a short kernel can be delayed behind a large one that already occupies the device. An execution context binds work to a fixed set of CUs: any kernel on a stream belonging to the context is confined to those CUs, and no kernel source changes are needed. This is HIP's counterpart to CUDA green contexts.

This example runs a fixed latency-sensitive workload against a saturated device and sweeps how many CUs the workload gets to itself, so the effect of partitioning is visible as a trend. In every case a long-running background kernel is launched to occupy the device, and a shorter critical kernel is timed while the background kernel runs:

1. **Baseline (shared CUs)**: the two kernels run on ordinary streams and compete for all of the device's CUs, so the critical kernel waits behind the background kernel.
2. **Partitioned (own CUs)**: the CUs are split into two execution contexts, the background kernel is confined to the larger group, and the critical kernel runs on its own group. This is repeated for a few partition sizes (roughly an eighth, a quarter, and half of the device).

The program prints a table of the critical kernel's latency and its speedup over the baseline for each configuration. As the critical partition grows, its latency drops well below the contended baseline.

Execution context resource partitioning is an AMD (HIP) feature. On the CUDA backend, where the required runtime support may be unavailable, only the shared-CU baseline runs; the partitioned sweep is guarded by `__HIP_PLATFORM_AMD__`.

### Application flow

1. The device is selected with `hipSetDevice`.
2. The number of compute units is determined. On the HIP (AMD) backend it comes from `hipDeviceGetDevResource` using the `hipDevResourceTypeSm` resource type. (The field is named `smCount` for CUDA source compatibility; on AMD GPUs it represents compute units.) On other backends it is read from `hipGetDeviceProperties`.
3. **Baseline.** Two ordinary non-blocking streams are created with `hipStreamCreateWithFlags`. The background kernel is launched on one, and the critical kernel is launched and timed on the other with HIP events while the background kernel runs. Both share all CUs.
4. **Partitioned sweep (HIP backend only).** For each candidate partition size, the CU resources are split into two groups with `hipDevSmResourceSplit`: a larger group for the background kernel and a smaller group dedicated to the critical kernel.
5. A resource descriptor is generated for each group with `hipDevResourceGenerateDesc`.
6. An execution context is created from each descriptor with `hipGreenCtxCreate`.
7. A stream is created for each execution context with `hipExecutionCtxStreamCreate`, and the same background-plus-critical timing is repeated. The critical kernel runs on its own partitioned CUs. The contexts and streams are then destroyed with `hipExecutionCtxDestroy` and `hipStreamDestroy` before the next partition size.
8. Each configuration's critical-kernel latency and speedup over the baseline are printed as a row in the results table.
9. The device output buffers are freed with `hipFree`.

## Key APIs and Concepts

Execution contexts carve a GPU's CUs into separate slices within one process, so urgent work has resources ready instead of waiting for a busy device to free up. Setting one up is a four-step sequence: read the device resources, split the CU resource, wrap the pieces in a descriptor, and create the context from it. A stream created on the context keeps every kernel launched on it inside that context's CUs. Sweeping the partition size while the background kernel saturates the device shows the critical kernel's latency fall as it gets more dedicated CUs.

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
