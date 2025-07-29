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

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <iostream>

#define ROUND_UP(SIZE,GRANULARITY) ((1 + SIZE / GRANULARITY) * GRANULARITY)

#define HIP_CHECK(expression)              \
{                                          \
    const hipError_t err = expression;     \
    if(err != hipSuccess)                  \
    {                                      \
        std::cerr << "HIP error: "         \
            << hipGetErrorString(err)      \
            << " at " << __LINE__ << "\n"; \
    }                                      \
}

__global__ void zeroAddr(int* pointer)
{
    *pointer = 0;
}

__global__ void fillAddr(int* pointer)
{
    *pointer = 42;
}


int main()
{
    int currentDev = 0;

    // Step 1: Check virtual memory management support on device 0
    int vmm = 0;
    HIP_CHECK(
        hipDeviceGetAttribute(
            &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
        )
    );

    std::cout << "Virtual memory management support value: " << vmm << std::endl;

    if (vmm == 0)
    {
        std::cout << "GPU 0 doesn't support virtual memory management.";
        return 0;
    }

    // Size of memory to allocate
    std::size_t size = 4 * 1024;

    // Step 2: Allocate physical memory
    hipMemGenericAllocationHandle_t allocHandle;
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = currentDev;
    std::size_t granularity = 0;
    HIP_CHECK(
        hipMemGetAllocationGranularity(
            &granularity,
            &prop,
            hipMemAllocationGranularityMinimum));
    std::size_t padded_size = ROUND_UP(size, granularity);
    HIP_CHECK(hipMemCreate(&allocHandle, padded_size * 2, &prop, 0));

    // Step 3: Reserve a virtual memory address range
    void* virtualPointer = nullptr;
    HIP_CHECK(hipMemAddressReserve(&virtualPointer, padded_size, granularity, nullptr, 0));

    // Step 4: Map the physical memory to the virtual address range
    HIP_CHECK(hipMemMap(virtualPointer, padded_size, 0, allocHandle, 0));

    // Step 5: Set memory access permission for pointer
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = currentDev;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    HIP_CHECK(hipMemSetAccess(virtualPointer, padded_size, &accessDesc, 1));

    // Step 6: Perform memory operation
    int value = 42;
    HIP_CHECK(hipMemcpy(virtualPointer, &value, sizeof(int), hipMemcpyHostToDevice));

    int result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 42)
    {
        std::cout << "Success. Value: " << result << std::endl;
    }
    else
    {
        std::cout << "Failure. Value: " << result << std::endl;
    }

    // Step 7: Launch kernels
    // Launch zeroAddr kernel
    zeroAddr<<<1, 1>>>((int*)virtualPointer);
    HIP_CHECK(hipDeviceSynchronize());

    // Check zeroAddr kernel result
    result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 0)
    {
        std::cout << "Success. zeroAddr kernel: " << result << std::endl;
    }
    else
    {
        std::cout << "Failure. zeroAddr kernel: " << result << std::endl;
    }

    // Launch fillAddr kernel
    fillAddr<<<1, 1>>>((int*)virtualPointer);
    HIP_CHECK(hipDeviceSynchronize());

    // Check fillAddr kernel result
    result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 42)
    {
        std::cout << "Success. fillAddr kernel: " << result << std::endl;
    }
    else
    {
        std::cout << "Failure. fillAddr kernel: " << result << std::endl;
    }

    // Step 8: Cleanup
    HIP_CHECK(hipMemUnmap(virtualPointer, padded_size));
    HIP_CHECK(hipMemRelease(allocHandle));
    HIP_CHECK(hipMemAddressFree(virtualPointer, padded_size));

    return 0;
}
// [sphinx-end]
