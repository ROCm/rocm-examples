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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>

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

// Declare function pointer
using hipInit_t = std::add_pointer_t<hipError_t(unsigned int)>;

int main()
{
    // Initialize the HIP runtime
    if (auto err = hipInit(0); err != hipSuccess)
    {
        std::cerr << "Failed to initialize HIP runtime." << std::endl;
        return EXIT_FAILURE;
    }

    // Get the address of the hipInit function
    hipInit_t hipInitFunc;
    int hipVersion = HIP_VERSION; // Use the HIP version defined in hip_runtime_api.h (included by hip_runtime.h)
    std::uint64_t flags = 0; // No special flags
    hipDriverProcAddressQueryResult symbolStatus;

    if (auto err = hipGetProcAddress("hipInit", reinterpret_cast<void**>(&hipInitFunc), hipVersion, flags, &symbolStatus);
        err != hipSuccess)
    {
        std::cerr << "Failed to get address of hipInit()." << std::endl;
        return EXIT_FAILURE;
    }

    // Call the hipInit function using the obtained address
    if(auto err = hipInitFunc(0); err != hipSuccess)
    {
        std::cerr << "Failed to initialize HIP runtime using hipGetProcAddress()." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "HIP runtime initialized successfully using hipGetProcAddress()." << std::endl;
    return EXIT_SUCCESS;
}
// [sphinx-end]


// The above also works for features not present at the time of writing the application. The below shows how to load
// a hypothetical function "foo" from a future HIP runtime.
using foo_t = std::add_pointer_t<hipError_t(unsigned int)>;

[[maybe_unused]] void hypothetical()
{
    // [sphinx-future-start]
    // Get the address of the foo function
    foo_t fooFunc;
    int hipVersion = 70300000; // Use an own HIP version number (e.g. 7.3.0)
    std::uint64_t flags = 0; // No special flags
    hipDriverProcAddressQueryResult symbolStatus;

    auto err = hipGetProcAddress("foo", (void**)&fooFunc, hipVersion, flags, &symbolStatus);
    // [sphinx-future-end]
    if(err != hipSuccess)
        std::cout << "foo() not yet present!" << std::endl;
}
