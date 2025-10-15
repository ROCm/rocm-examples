# HIP-Doc HIP Graph API tutorial

## Description

This example demonstrates how to port an existing stream-based application to a graph-based application. For more
information on this topic, please refer to the
[HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html).

### Prerequisites

On Windows, this example can only be built by CMake and the Ninja generator. Visual Studio generators are not supported.

### Application flow

#### Stream variant

1. The number of asynchronous engines on the device is queried.
2. A phantom volume is created and forward projections are created from it.
3. The required data buffers, hipFFT plans, etc. are set up.
4. The forward projections are processed in batches. The batch size is equal to the number of asynchronous engines. Each
   batch item is processed in its own stream. Each projection is processed in the following way:
    1. All pixels in the projection are normalized, i.e. transformed from a 12-bit unsigned integer value to a 32-bit
       floating-point value in the range [0, 1].
    2. The projection is log transformed.
    3. The projection is weighted.
    4. The projection is transformed into Fourier space.
    5. The projection is filtered in Fourier space.
    6. The filtered projection is transformed into real space.
    7. The filtered projection is back-projected.
5. The resulting volume is saved to disk.
6. The data is cleaned up.

#### Capturing variant

1. The number of asynchronous engines on the device is queried.
2. A phantom volume is created and forward projections are created from it.
3. The required data buffers, hipFFT plans, etc. are set up.
4. The forward projections are processed in batches. The batch size is equal to the number of asynchronous engines. Each
   batch item is processed in its own stream. All processing streams are captured into a HIP graph template.
   Each projection is processed in the following way:
    1. All pixels in the projection are normalized, i.e. transformed from a 12-bit unsigned integer value to a 32-bit
       floating-point value in the range [0, 1].
    2. The projection is log transformed.
    3. The projection is weighted.
    4. The projection is transformed into Fourier space.
    5. The projection is filtered in Fourier space.
    6. The filtered projection is transformed into real space.
    7. The filtered projection is back-projected.
5. The resulting graph template is instantiated and launched on a stream.
6. The resulting volume is saved to disk.
7. The data is cleaned up.

#### Manual creation variant

1. The number of asynchronous engines on the device is queried.
2. A phantom volume is created and forward projections are created from it.
3. The required data buffers, hipFFT plans, etc. are set up.
4. The forward projections are processed in batches. The batch size is equal to the number of asynchronous engines. Each
   batch item represents a branch in the graph template, and each branch consists of the following steps where each step
   represents a node in the graph:
    1. All pixels in the projection are normalized, i.e. transformed from a 12-bit unsigned integer value to a 32-bit
       floating-point value in the range [0, 1].
    2. The projection is log transformed.
    3. The projection is weighted.
    4. The projection is transformed into Fourier space.
    5. The projection is filtered in Fourier space.
    6. The filtered projection is transformed into real space.
    7. The filtered projection is back-projected.
5. The resulting graph template is instantiated and launched on a stream.
6. The resulting volume is saved to disk.
7. The data is cleaned up.

## Demonstrated API calls

### HIP Runtime

#### Device symbols

* `atomicAdd`
* `blockIdx`
* `blockDim`
* `cosf`
* `expf`
* `fabsf`
* `floorf`
* `fmaxf`
* `fminf`
* `logf`
* `max`
* `min`
* `roundf`
* `sinf`
* `sqrtf`
* `tex2D`
* `threadIdx`

#### Host symbols

* `hipCreateChannelDesc`
* `hipCreateTextureObject`
* `hipDestroyTextureObject`
* `hipDeviceSynchronize`
* `hipEventCreate`
* `hipEventDestroy`
* `hipEventRecord`
* `hipFree`
* `hipFreeHost`
* `hipGetDeviceProperties`
* `hipGetErrorString`
* `hipGraphAddKernelNode`
* `hipGraphAddMemcpyNode`
* `hipGraphAddMemsetNode`
* `hipGraphCreate`
* `hipGraphDebugDotPrint`
* `hipGraphDestroy`
* `hipGraphExecDestroy`
* `hipGraphExecUpdate`
* `hipGraphGetNodes`
* `hipGraphInstantiate`
* `hipGraphLaunch`
* `hipGraphNodeGetDependentNodes`
* `hipHostMalloc`
* `hipLaunchHostFunc`
* `hipMalloc`
* `hipMalloc3D`
* `hipMallocPitch`
* `hipMemcpy2DAsync`
* `hipMemcpy3DAsync`
* `hipMemGetInfo`
* `hipMemset2DAsync`
* `hipMemset3D`
* `hipStreamBeginCapture`
* `hipStreamBeginCaptureToGraph`
* `hipStreamCreate`
* `hipStreamEndCapture`
* `hipStreamDestroy`
* `hipStreamSynchronize`
* `hipStreamWaitEvent`
* `make_hipExtent`
* `make_hipPitchedPtr`
* `make_hipPos`

### hipFFT library symbols

* `hipfftCreate`
* `hipfftExecC2R`
* `hipfftExecR2C`
* `hipfftDestroy`
* `hipfftMakePlanMany`
* `hipfftPlan1d`
* `hipfftSetStream`
