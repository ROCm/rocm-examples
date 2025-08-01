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

#include <cstdlib>
#include <fstream>
#include <iostream>

int main()
{
    // Considering that you have exported the memory pool pointer already.
    // Now, let's simulate reading the exported data from a named pipe (FIFO).
    const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
    std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

    if (!fifoStream.is_open())
    {
        std::cerr << "Error opening FIFO file: " << fifoPath << std::endl;
        return EXIT_FAILURE;
    }

    // Read the exported data.
    hipMemPoolPtrExportData importData;
    fifoStream.read(reinterpret_cast<char*>(&importData), sizeof(hipMemPoolPtrExportData));
    fifoStream.close();

    if (fifoStream.fail())
    {
        std::cerr << "Error reading from FIFO file." << std::endl;
        return EXIT_FAILURE;
    }

    // Create a memory pool with default properties.
    hipMemPoolProps poolProps = {};
    poolProps.allocType = hipMemAllocationTypePinned;
    poolProps.handleTypes = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type = hipMemLocationTypeDevice;
    poolProps.location.id = 0; // Assuming device 0.

    hipMemPool_t memPool;
    hipMemPoolCreate(&memPool, &poolProps);

    // Import the memory pool pointer.
    void* importedDevPtr;
    hipError_t result = hipMemPoolImportPointer(&importedDevPtr, memPool, &importData);
    if (result != hipSuccess)
    {
        std::cerr << "Error imported memory pool pointer: " << hipGetErrorString(result) << std::endl;
        return EXIT_FAILURE;
    }

    // Now you can use the importedDevPtr for your computations.

    // Clean up (free the memory).
    hipFree(importedDevPtr);
    hipMemPoolDestroy(memPool);

    return EXIT_SUCCESS;
}
// [sphinx-end]
