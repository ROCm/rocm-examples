#ifndef _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP
#define _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP

#include "glad/glad.h"

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

// TODO: Remove this once HIP supports these symbols.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/49.
#if defined(__HIP_PLATFORM_NVIDIA__) && !defined(hipGLDeviceListAll) && HIP_VERSION_MAJOR < 6

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

#endif

#endif
