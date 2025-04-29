// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _HIP_BASIC_VULKAN_TEXTURE_INTEROP_NVIDIA_HIP_FIX_HPP
#define _HIP_BASIC_VULKAN_TEXTURE_INTEROP_NVIDIA_HIP_FIX_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

// Currently these HIP symbols are missing when compiling for NVIDIA.
// TODO: Remove this once HIP supports these symbols.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/49.
#if defined(__HIP_PLATFORM_NVIDIA__) && !defined(hipExternalMemoryHandleTypeOpaqueFd) \
    && HIP_VERSION_MAJOR < 6
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
    #define hipExternalMemoryHandleTypeOpaqueWin32Kmt cudaExternalMemoryHandleTypeOpaqueWin32Kmt
    #define hipExternalSemaphoreHandleTypeOpaqueWin32 cudaExternalSemaphoreHandleTypeOpaqueWin32

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
