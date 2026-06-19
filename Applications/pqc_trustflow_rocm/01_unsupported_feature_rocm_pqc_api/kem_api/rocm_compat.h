#ifndef ROCM_COMPAT_H
#define ROCM_COMPAT_H

#if defined(USE_HIP) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#ifndef USE_HIP
#define USE_HIP 1
#endif

#define GPU_USE_HIP 1

#include <hip/hip_runtime.h>

#define GPU_RUNTIME_NAME "HIP"

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaLimitStackSize hipLimitStackSize
#define cudaFuncSetCacheConfig hipFuncSetCacheConfig
#define cudaFuncCachePreferL1 hipFuncCachePreferL1

#else

#define GPU_USE_HIP 0

#include <cuda_runtime.h>

#define GPU_RUNTIME_NAME "CUDA"

#endif

#endif /* ROCM_COMPAT_H */
