/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef COMMON_RCCL_UTILS_HPP
#define COMMON_RCCL_UTILS_HPP

#include "example_utils.hpp"

#include <rccl/rccl.h>

#include <iostream>

/// \brief Converts a \p ncclResult_t variable to its correspondent string.
inline const char* nccl_result_to_string(ncclResult_t result)
{
    switch(result)
    {
        case ncclSuccess: return "ncclSuccess";
        case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
        case ncclSystemError: return "ncclSystemError";
        case ncclInternalError: return "ncclInternalError";
        case ncclInvalidArgument: return "ncclInvalidArgument";
        case ncclInvalidUsage: return "ncclInvalidUsage";
        case ncclRemoteError: return "ncclRemoteError";
        case ncclInProgress: return "ncclInProgress";
        case ncclNumResults: return "ncclNumResults";
        default: return "<unknown ncclResult_t value>";
    }
}

/// \brief Checks if the provided status code is \p ncclSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define RCCL_CHECK(condition)                                                                    \
    {                                                                                            \
        const ncclResult_t result = (condition);                                                 \
        if(result != ncclSuccess)                                                                \
        {                                                                                        \
            std::cerr << "RCCL error encountered: \"" << nccl_result_to_string(result) << "\" (" \
                      << ncclGetErrorString(result) << ")"                                       \
                      << " at " << __FILE__ << ':' << __LINE__ << std::endl;                     \
            std::exit(error_exit_code);                                                          \
        }                                                                                        \
    }

/// \brief Detect the number of available GPUs in the system
inline int detect_num_gpus()
{
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    return device_count;
}

/// \brief Print information about a specific GPU device
inline void print_gpu_info(int device_id)
{
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::cout << "GPU " << device_id << ": " << props.name << " (Compute " << props.major << "."
              << props.minor << ")" << std::endl;
}

#endif // COMMON_RCCL_UTILS_HPP
