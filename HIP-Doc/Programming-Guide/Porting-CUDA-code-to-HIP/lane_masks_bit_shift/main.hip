// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/hipcub.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if (err != hipSuccess)                                   \
    {                                                        \
        std::cout << "HIP Error: " << hipGetErrorString(err) \
              << " at line " << __LINE__ << std::endl;       \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}

// Problematic kernel: Uses 32-bit integer for lane masks, which fails on warp sizes > 32
__global__ void initial_problematic(std::uint32_t* results)
{
    using WarpReduce = hipcub::WarpReduce<std::uint32_t>;
    __shared__ typename WarpReduce::TempStorage tempStorage;
    
    const unsigned int tid = threadIdx.x;
    unsigned int val = tid % warpSize;

    // [sphinx-initial-start]
    // Get the thread's position in the warp
    unsigned int laneId = threadIdx.x % warpSize;

    // Use lane ID for bit-shift - PROBLEM: 1 is a 32-bit literal
    // On AMD GPUs with warpSize=64, this can overflow when laneId >= 32
    std::uint32_t mask = (1 << laneId) - 1;

    // Shift 32 bit integer with val variable - PROBLEM: can overflow
    std::uint32_t shift_result = (val < warpSize) ? (1 << val) : 0;
    std::uint32_t warp_sum = WarpReduce(tempStorage).Sum(shift_result);
    // [sphinx-initial-end]

    // Store result for the first thread in each warp
    if (laneId == 0)
        results[tid / warpSize] = warp_sum;
}

// Improved kernel: Uses 64-bit integers explicitly
__global__ void improved_64bit(std::uint64_t* results)
{
    using WarpReduce = hipcub::WarpReduce<std::uint64_t>;
    __shared__ typename WarpReduce::TempStorage tempStorage;

    const unsigned int tid = threadIdx.x;
    unsigned int val = tid % warpSize;

    // [sphinx-improved-start]
    // Get the thread's position in the warp
    unsigned int laneId = threadIdx.x % warpSize;

    // Use lane ID for bit-shift - FIXED: 1ull is a 64-bit literal
    std::uint64_t mask = (1ull << laneId) - 1;

    // Shift 64 bit integer with val variable - FIXED: works for warpSize up to 64
    std::uint64_t shift_result = (val < warpSize) ? (1ull << val) : 0;
    std::uint64_t warp_sum = WarpReduce(tempStorage).Sum(shift_result);
    // [sphinx-improved-end]

    // Store result for the first thread in each warp
    if (laneId == 0)
        results[tid / warpSize] = warp_sum;
}

// [sphinx-typedef-start]
// Portable lane mask type: adjusts based on GPU architecture
#ifdef __GFX9__
using lane_mask_t = std::uint64_t;
#else
using lane_mask_t = std::uint32_t;
#endif
// [sphinx-typedef-end]

// Best practice kernel: Uses portable lane_mask_t type
__global__ void best_portable(lane_mask_t* results)
{
    using WarpReduce = hipcub::WarpReduce<lane_mask_t>;
    __shared__ typename WarpReduce::TempStorage tempStorage;

    const unsigned int tid = threadIdx.x;
    unsigned int val = tid % warpSize;

    // [sphinx-best-start]
    // Get the thread's position in the warp
    unsigned int laneId = threadIdx.x % warpSize;

    // Use lane ID for bit-shift - PORTABLE: lane_mask_t adjusts to warp size
    lane_mask_t mask = (lane_mask_t{1} << laneId) - 1;

    // Shift 32 or 64 bit integer with val variable - PORTABLE
    lane_mask_t shift_result = (val < warpSize) ? (lane_mask_t{1} << val) : 0;
    lane_mask_t warp_sum = WarpReduce(tempStorage).Sum(shift_result);
    // [sphinx-best-end]

    // Store result for the first thread in each warp
    if (laneId == 0)
        results[tid / warpSize] = warp_sum;
}

int main()
{
    // Query device properties
    int warpSizeHost;
    HIP_CHECK(hipDeviceGetAttribute(&warpSizeHost, hipDeviceAttributeWarpSize, 0));
    
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
    
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Warp size: " << warpSizeHost << std::endl;
    std::cout << std::endl;

    const int num_threads = 128;
    const int num_warps = num_threads / warpSizeHost;

    // Test 1: Problematic kernel (only safe for warpSize <= 32)
    if (warpSizeHost <= 32)
    {
        std::cout << "=== Test 1: Problematic 32-bit Implementation ===" << std::endl;
        std::cout << "Running on warpSize=" << warpSizeHost 
                  << " (safe for this architecture)" << std::endl;
        
        std::uint32_t* d_results_32;
        HIP_CHECK(hipMalloc(&d_results_32, num_warps * sizeof(std::uint32_t)));
        
        initial_problematic<<<1, num_threads>>>(d_results_32);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        std::vector<std::uint32_t> h_results_32;
        h_results_32.resize(num_warps);
        HIP_CHECK(hipMemcpy(h_results_32.data(), d_results_32, 
                           num_warps * sizeof(std::uint32_t), 
                           hipMemcpyDeviceToHost));
        
        std::cout << "Results: ";
        for (int i = 0; i < num_warps; ++i)
        {
            std::cout << h_results_32[i] << " ";
        }
        std::cout << std::endl;
        
        HIP_CHECK(hipFree(d_results_32));
    }
    else
    {
        std::cout << "=== Test 1: Skipped ===" << std::endl;
        std::cout << "Problematic 32-bit kernel skipped (warpSize=" << warpSizeHost 
                  << " would cause overflow)" << std::endl;
    }
    std::cout << std::endl;

    // Test 2: Improved 64-bit implementation
    std::cout << "=== Test 2: Improved 64-bit Implementation ===" << std::endl;
    std::uint64_t* d_results_64;
    HIP_CHECK(hipMalloc(&d_results_64, num_warps * sizeof(std::uint64_t)));
    
    improved_64bit<<<1, num_threads>>>(d_results_64);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    std::vector<std::uint64_t> h_results_64;
    h_results_64.resize(num_warps);
    HIP_CHECK(hipMemcpy(h_results_64.data(), d_results_64, 
                       num_warps * sizeof(std::uint64_t), 
                       hipMemcpyDeviceToHost));
    
    std::cout << "Results: ";
    for (int i = 0; i < num_warps; ++i)
    {
        std::cout << h_results_64[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    HIP_CHECK(hipFree(d_results_64));

    // Test 3: Best practice portable implementation
    std::cout << "=== Test 3: Best Practice Portable Implementation ===" << std::endl;
    std::cout << "Using lane_mask_t (";
    std::cout << (sizeof(lane_mask_t) == 8 ? "64-bit" : "32-bit");
    std::cout << " on this architecture)" << std::endl;
    
    lane_mask_t* d_results_portable;
    HIP_CHECK(hipMalloc(&d_results_portable, num_warps * sizeof(lane_mask_t)));
    
    best_portable<<<1, num_threads>>>(d_results_portable);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    std::vector<lane_mask_t> h_results_portable;
    h_results_portable.resize(num_warps);
    HIP_CHECK(hipMemcpy(h_results_portable.data(), d_results_portable,
                       num_warps * sizeof(lane_mask_t), 
                       hipMemcpyDeviceToHost));
    
    std::cout << "Results: ";
    for (int i = 0; i < num_warps; ++i)
    {
        std::cout << h_results_portable[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    HIP_CHECK(hipFree(d_results_portable));

    // Verify results match between 64-bit and portable versions
    bool results_match = true;
    for (int i = 0; i < num_warps; ++i)
    {
        if (h_results_64[i] != static_cast<std::uint64_t>(h_results_portable[i]))
        {
            results_match = false;
            break;
        }
    }
    
    std::cout << "=== Verification ===" << std::endl;
    std::cout << "64-bit and portable results match: " 
              << (results_match ? "YES" : "NO") << std::endl;

    if (results_match)
    {
        std::cout << "\nSUCCESS: All kernels produced correct results!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "\nFAILURE: Results do not match!" << std::endl;
        return EXIT_FAILURE;
    }
}
