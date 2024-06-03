// MIT License
//
// Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <iomanip>
#include <iostream>

#include <hip/hip_runtime.h>

namespace
{
/// Number of characters in the first column.
constexpr unsigned int col_w = 26;

double khz_to_mhz(size_t f)
{
    return f / 1000.;
}

double bytes_to_kib(size_t s)
{
    return s / 1024.;
}

double bytes_to_gib(size_t s)
{
    return s / (1024. * 1024. * 1024.);
}

/// Print properties of device \p device_id.
void print_device_properties(int device_id)
{
    std::cout
        << std::setw(col_w)
        << "--------------------------------------------------------------------------------\n";
    std::cout << std::setw(col_w) << "Device ID " << device_id << '\n';
    std::cout
        << std::setw(col_w)
        << "--------------------------------------------------------------------------------\n";

    // Query the device properties.
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    // Print a small set of all available properties. A full list can be found at:
    // https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html
    std::cout << std::setw(col_w) << "Name: " << props.name << '\n';
    std::cout << std::setw(col_w)
              << "totalGlobalMem: " << double_precision(bytes_to_gib(props.totalGlobalMem), 2, true)
              << " GiB\n";
    std::cout << std::setw(col_w) << "sharedMemPerBlock: " << bytes_to_kib(props.sharedMemPerBlock)
              << " KiB\n";
    std::cout << std::setw(col_w) << "regsPerBlock: " << props.regsPerBlock << '\n';
    std::cout << std::setw(col_w) << "warpSize: " << props.warpSize << '\n';
    std::cout << std::setw(col_w) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << '\n';
    std::cout << std::setw(col_w) << "maxThreadsDim: "
              << "(" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", "
              << props.maxThreadsDim[2] << ")\n";
    std::cout << std::setw(col_w) << "maxGridSize: "
              << "(" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", "
              << props.maxGridSize[2] << ")\n";
    std::cout << std::setw(col_w) << "clockRate: " << khz_to_mhz(props.clockRate) << " Mhz\n";

    // On the AMD platform only, print the architecture name.
#ifdef __HIP_PLATFORM_AMD__
    std::cout << std::setw(col_w) << "gcnArchName: " << props.gcnArchName << '\n';
#elif defined(__HIP_PLATFORM_NVIDIA__)
    std::cout << std::setw(col_w) << "major.minor: " << props.major << "." << props.minor << '\n';
#endif
    std::cout << '\n';

    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    int device_peer_count = 0;
    std::cout << std::setw(col_w) << "peer(s): ";
    for(int i = 0; i < device_count; i++)
    {
        if(i == device_id)
        {
            continue;
        }
        int is_peer;
        HIP_CHECK(hipDeviceCanAccessPeer(&is_peer, i, device_id));
        if(is_peer == 1)
        {
            if(device_peer_count > 0)
            {
                std::cout << ", ";
            }
            device_peer_count++;
            std::cout << "Device " << i << " ";
        }
    }
    if(device_peer_count == 0)
    {
        std::cout << "None";
    }
    std::cout << '\n';

    // On the NVIDIA platform, print CUDA-specific code.
#ifdef __HIP_PLATFORM_NVIDIA__
    std::cout << '\n';

    size_t val;
    cudaDeviceGetLimit(&val, cudaLimitStackSize);
    std::cout << std::setw(col_w) << "cudaLimitStackSize: " << val << " bytes/thread\n";
    cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize);
    std::cout << std::setw(col_w) << "cudaLimitMallocHeapSize: " << bytes_to_kib(val)
              << " KiB/device\n";
#endif
    std::cout << '\n';

    size_t free, total;
    HIP_CHECK(hipMemGetInfo(&free, &total));

    std::cout << std::setw(col_w)
              << "memInfo.total: " << double_precision(bytes_to_gib(total), 2, true) << " GiB\n";
    std::cout << std::setw(col_w)
              << "memInfo.free:  " << double_precision(bytes_to_gib(free), 2, true) << " GiB ("
              << double_precision(static_cast<double>(free) / total * 100.0, 0, true) << "%)\n";
}
} // namespace

int main()
{
    std::cout << std::left;

    // Identify HIP target platform.
    std::cout << std::setw(col_w) << "Hip target platform: ";
#if defined(__HIP_PLATFORM_AMD__)
    std::cout << "AMD";
#elif defined(__HIP_PLATFORM_NVIDIA__)
    std::cout << "NVIDIA";
#endif
    std::cout << '\n';

    // Identify compiler.
    std::cout << std::setw(col_w) << "Compiler: ";
#if defined(__HIP_PLATFORM_AMD__)
    std::cout << "HIP-Clang";
#elif defined(__HIP_PLATFORM_NVIDIA__) && defined(__CUDACC__)
    std::cout << "nvcc (CUDA language extensions enabled)";
#elif defined(__HIP_PLATFORM_NVIDIA__) && !defined(__CUDACC__)
    std::cout << "nvcc (pass-through mode to an underlying host compiler)";
#endif
    std::cout << '\n';

    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    for(int i = 0; i < device_count; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        print_device_properties(i);
    }
}
