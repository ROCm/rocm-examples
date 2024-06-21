# HIP-Basic Texture Management Example

## Description

This example demonstrates how a kernel may use texture memory through the texture object API. Using texture memory may be beneficial as the texture cache is optimized for 2D spatial locality and exposes features such as hardware filtering. In the example, a texture is created using a device array and is sampled in a kernel to create a histogram of its values.

### Application flow

1. Check whether texture functions are supported on the device.
2. Initialize the texture data on host side.
3. Specify the channel description of the texture and allocate a device array based on the texture dimensions and channel descriptor.
4. Copy the texture data from host to device.
5. Specify the texture resource and its parameters, and create the texture object.
6. Allocate a device-side histogram.
7. Launch the histogram kernel, which creates a histogram of the texture on the device.
8. Copy the histogram to host memory and print the results.
9. Destroy the texture object and release resources.

## Key APIs and Concepts

- The memory for the texture may be a device array `hipArray_t`, which is allocated with `hipMallocArray`. The allocation call requires a channel descriptor `hipChannelFormatDesc` and the dimensions of the texture. The channel descriptor can be created using `hipCreateChannelDesc`. Host data can be transferred to the device array using `hipMemcpy2DToArray`.
- The texture object `hipTextureObject_t` is created with `hipCreateTextureObject`, which requires a resource descriptor `hipResourceDesc` and a texture descriptor `hipTextureDesc`. The resource descriptor describes the resource used to create the texture, in this example a device array `hipResourceTypeArray`. The texture descriptor describes the properties of the texture, such as its addressing mode and whether it uses normalized coordinates.
- The created texture object can be sampled in a kernel using `tex2D`.
- The texture object is cleaned up by calling `hipDestroyTextureObject` and the device array is cleaned up by calling `hipFreeArray`.

## Demonstrated API Calls

### HIP runtime

- `__global__`

#### Device symbols

- `atomicAdd`
- `blockDim`
- `blockIdx`
- `tex2D`
- `threadIdx`

#### Host symbols

- `hipArray_t`
- `hipAddressModeWrap`
- `hipChannelFormatDesc`
- `hipChannelFormatKindUnsigned`
- `hipCreateChannelDesc`
- `hipCreateTextureObject`
- `hipDestroyTextureObject`
- `hipDeviceAttributeImageSupport`
- `hipDeviceGetAttribute`
- `hipFilterModePoint`
- `hipFree`
- `hipFreeArray`
- `hipGetLastError`
- `hipMalloc`
- `hipMallocArray`
- `hipMemcpy`
- `hipMemcpy2DToArray`
- `hipMemcpyHostToDevice`
- `hipMemset`
- `hipReadModeElementType`
- `hipResourceDesc`
- `hipResourceTypeArray`
- `hipStreamDefault`
- `hipTextureDesc`
- `hipTextureObject_t`

## Limitations

This example is not supported on CDNA3 architecture (MI300) and above.
