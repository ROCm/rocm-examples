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

#define HIP_CHECK(expression)                        \
{                                                    \
    const hipError_t status = expression;            \
    if (status != hipSuccess)                        \
    {                                                \
        std::cerr << "HIP error " << status          \
                << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;            \
        std::exit(EXIT_FAILURE);                     \
    }                                                \
}

int main()
{
    // Considering that you have exported the memory pool pointer already.
    // Now, let's simulate reading the exported data from a named pipe (FIFO).
    const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path
    std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

    if (!fifoStream.is_open())
    {
        std::cerr << "Error opening FIFO file: " << fifoPath << std::endl;
        return EXIT_FAILURE;
    }

    // Read the exported data.
    int descriptor;
    fifoStream.read(reinterpret_cast<char*>(&descriptor), sizeof(int));
    fifoStream.close();

    if (fifoStream.fail())
    {
        std::cerr << "Error reading from FIFO file." << std::endl;
        return EXIT_FAILURE;
    }

    // Import the memory pool.
    hipMemPool_t memPool;
    hipError_t result = hipMemPoolImportFromShareableHandle(&memPool, &descriptor, hipMemHandleTypePosixFileDescriptor, 0);
    if (result != hipSuccess)
    {
        std::cerr << "Error importing memory pool: " << hipGetErrorString(result) << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory from the imported memory pool.
    void* importedDevPtr;
    HIP_CHECK(hipMallocFromPoolAsync(&importedDevPtr, sizeof(int), memPool, 0));

    // Now you can use the importedDevPtr for your computations.

    // Clean up (free the memory).
    HIP_CHECK(hipFree(importedDevPtr));
    HIP_CHECK(hipMemPoolDestroy(memPool));

    return EXIT_SUCCESS;
}
// [sphinx-end]
