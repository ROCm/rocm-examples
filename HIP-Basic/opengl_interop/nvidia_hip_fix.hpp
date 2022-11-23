#ifndef _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP
#define _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP

#include "glad/glad.h"

#include <hip/hip_runtime.h>

// TODO: Remove this once HIP supports these symbols.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/49.
#if defined(__HIP_PLATFORM_NVCC__) && !defined(hipGLDeviceListAll)

    #include <cuda_gl_interop.h>

    #define hipGLDeviceListAll cudaGLDeviceListAll
    #define hipGLDeviceList cudaGLDeviceList
    #define hipGraphicsResource_t cudaGraphicsResource_t
    #define hipGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsWriteDiscard

hipError_t hipGLGetDevices(unsigned int* const   pHipDeviceCount,
                           int* const            pHipDevices,
                           const unsigned int    hipDeviceCount,
                           const hipGLDeviceList deviceList)
{
    return hipCUDAErrorTohipError(
        cudaGLGetDevices(pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList));
}

hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource_t* const resource,
                                       const GLuint                 buffer,
                                       const unsigned int           flags)
{
    return hipCUDAErrorTohipError(cudaGraphicsGLRegisterBuffer(resource, buffer, flags));
}

hipError_t hipGraphicsMapResources(const int                    count,
                                   hipGraphicsResource_t* const resources,
                                   const hipStream_t            stream = 0)
{
    return hipCUDAErrorTohipError(cudaGraphicsMapResources(count, resources, stream));
}

hipError_t hipGraphicsResourceGetMappedPointer(void** const                 dev_ptr,
                                               size_t* const                size,
                                               const cudaGraphicsResource_t resource)
{
    return hipCUDAErrorTohipError(cudaGraphicsResourceGetMappedPointer(dev_ptr, size, resource));
}

hipError_t hipGraphicsUnmapResources(const int                    count,
                                     hipGraphicsResource_t* const resources,
                                     const hipStream_t            stream = 0)
{
    return hipCUDAErrorTohipError(cudaGraphicsUnmapResources(count, resources, stream));
}

hipError_t hipGraphicsUnregisterResource(const hipGraphicsResource_t resource)
{
    return hipCUDAErrorTohipError(cudaGraphicsUnregisterResource(resource));
}

#endif

#endif
