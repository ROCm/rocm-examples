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
#include <cstdint>
#include <iostream>

// Sample helper functions for getting the usage statistics in bulk.
struct usageStatistics
{
    std::uint64_t reservedMemCurrent;
    std::uint64_t reservedMemHigh;
    std::uint64_t usedMemCurrent;
    std::uint64_t usedMemHigh;
};

void getUsageStatistics(hipMemPool_t memPool, struct usageStatistics *statistics)
{
    hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemCurrent, &statistics->reservedMemCurrent);
    hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &statistics->reservedMemHigh);
    hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemCurrent, &statistics->usedMemCurrent);
    hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &statistics->usedMemHigh);
}

// Resetting the watermarks resets them to the current value.
void resetStatistics(hipMemPool_t memPool)
{
    std::uint64_t value = 0;
    hipMemPoolSetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &value);
    hipMemPoolSetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &value);
}

int main()
{
    hipMemPool_t memPool;
    hipDevice_t device = 0; // Specify the device index.

    // Initialize the device.
    hipSetDevice(device);

    // Get the default memory pool for the device.
    hipDeviceGetDefaultMemPool(&memPool, device);

    // Allocate memory from the pool (e.g., 1 MB).
    std::size_t allocSize = 1 * 1024 * 1024;
    void* ptr;
    hipMalloc(&ptr, allocSize);

    // Free the allocated memory.
    hipFree(ptr);

    // Trim the memory pool to a specific size (e.g., 512 KB).
    std::size_t newSize = 512 * 1024;
    hipMemPoolTrimTo(memPool, newSize);

    // Get and print usage statistics before resetting.
    usageStatistics statsBefore;
    getUsageStatistics(memPool, &statsBefore);
    std::cout << "Before resetting statistics:" << std::endl;
    std::cout << "Reserved Memory Current: " << statsBefore.reservedMemCurrent << " bytes" << std::endl;
    std::cout << "Reserved Memory High: " << statsBefore.reservedMemHigh << " bytes" << std::endl;
    std::cout << "Used Memory Current: " << statsBefore.usedMemCurrent << " bytes" << std::endl;
    std::cout << "Used Memory High: " << statsBefore.usedMemHigh << " bytes" << std::endl;

    // Reset the statistics.
    resetStatistics(memPool);

    // Get and print usage statistics after resetting.
    usageStatistics statsAfter;
    getUsageStatistics(memPool, &statsAfter);
    std::cout << "After resetting statistics:" << std::endl;
    std::cout << "Reserved Memory Current: " << statsAfter.reservedMemCurrent << " bytes" << std::endl;
    std::cout << "Reserved Memory High: " << statsAfter.reservedMemHigh << " bytes" << std::endl;
    std::cout << "Used Memory Current: " << statsAfter.usedMemCurrent << " bytes" << std::endl;
    std::cout << "Used Memory High: " << statsAfter.usedMemHigh << " bytes" << std::endl;

    // Clean up.
    hipMemPoolDestroy(memPool);

    return 0;
}
// [sphinx-end]
