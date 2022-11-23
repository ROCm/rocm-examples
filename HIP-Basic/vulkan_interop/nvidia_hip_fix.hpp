#ifndef _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP
#define _HIP_BASIC_VULKAN_INTEROP_NVIDIA_HIP_FIX_HPP

#include <hip/hip_runtime.h>

// Currently these HIP symbols are missing when compiling for NVIDIA.
// TODO: Remove this once HIP supports these symbols.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/49.
#if defined(__HIP_PLATFORM_NVCC__) && !defined(hipExternalMemoryHandleTypeOpaqueFd)
    #define hipExternalMemoryHandleType cudaExternalMemoryHandleType
    #define hipExternalMemoryHandleTypeOpaqueFd cudaExternalMemoryHandleTypeOpaqueFd
    #define hipExternalSemaphoreHandleType cudaExternalSemaphoreHandleType
    #define hipExternalSemaphoreHandleTypeOpaqueFd cudaExternalSemaphoreHandleTypeOpaqueFd
    #define hipExternalMemory_t cudaExternalMemory_t
    #define hipExternalMemoryHandleDesc cudaExternalMemoryHandleDesc
    #define hipExternalSemaphoreHandleDesc cudaExternalSemaphoreHandleDesc
    #define hipExternalMemoryBufferDesc cudaExternalMemoryBufferDesc
    #define hipExternalSemaphore_t cudaExternalSemaphore_t
    #define hipExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParams
    #define hipExternalSemaphoreWaitParams cudaExternalSemaphoreWaitParams

hipError_t hipImportExternalMemory(hipExternalMemory_t* extmem, hipExternalMemoryHandleDesc* desc)
{
    return hipCUDAErrorTohipError(cudaImportExternalMemory(extmem, desc));
}

hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t*               extmem,
                                      const hipExternalSemaphoreHandleDesc* desc)
{
    return hipCUDAErrorTohipError(cudaImportExternalSemaphore(extmem, desc));
}

hipError_t hipDestroyExternalMemory(hipExternalMemory_t extmem)
{
    return hipCUDAErrorTohipError(cudaDestroyExternalMemory(extmem));
}

hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extmem)
{
    return hipCUDAErrorTohipError(cudaDestroyExternalSemaphore(extmem));
}

hipError_t hipExternalMemoryGetMappedBuffer(void**                       ptr,
                                            hipExternalMemory_t          extmem,
                                            hipExternalMemoryBufferDesc* desc)
{
    return hipCUDAErrorTohipError(cudaExternalMemoryGetMappedBuffer(ptr, extmem, desc));
}

hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t*           extsems,
                                            const hipExternalSemaphoreSignalParams* params,
                                            unsigned int                            num_sems,
                                            hipStream_t                             stream)
{
    return hipCUDAErrorTohipError(
        cudaSignalExternalSemaphoresAsync(extsems, params, num_sems, stream));
}

hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t*         extsems,
                                          const hipExternalSemaphoreWaitParams* params,
                                          unsigned int                          num_sems,
                                          hipStream_t                           stream)
{
    return hipCUDAErrorTohipError(
        cudaWaitExternalSemaphoresAsync(extsems, params, num_sems, stream));
}
#endif

#endif
