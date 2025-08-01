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

#include <sys/stat.h>

int main()
{
    // Allocate memory.
    void* devPtr;
    hipMalloc(&devPtr, sizeof(int));

    // Export the memory pool pointer.
    hipMemPoolPtrExportData exportData;
    hipError_t result = hipMemPoolExportPointer(&exportData, devPtr);
    if (result != hipSuccess)
    {
        std::cerr << "Error exporting memory pool pointer: " << hipGetErrorString(result) << std::endl;
        return EXIT_FAILURE;
    }

    // Create a named pipe (FIFO).
    const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
    mkfifo(fifoPath, 0666);

    // Write the exported data to the named pipe.
    std::ofstream fifoStream(fifoPath, std::ios::out | std::ios::binary);
    fifoStream.write(reinterpret_cast<char*>(&exportData), sizeof(hipMemPoolPtrExportData));
    fifoStream.close();

    // Clean up.
    hipFree(devPtr);

    return EXIT_SUCCESS;
}
// [sphinx-end]
