// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipfft_utils.hpp"
#include "hiprtc_utils.hpp"

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

#include <iostream>
#include <random>
#include <vector>

struct load_cbdata
{
    hipfftDoubleComplex* filter;
    double               scale;
};

static const char* load_callback_src = R"(
#ifdef __HIP_PLATFORM_AMD__
// rocFFT expects the provided symbol to be found verbatim in the
// compiled SPIR-V.  Give the callback function C linkage so that its
// actual name matches its apparent name in the source code.
#define CALLBACK_LINKAGE extern "C"
#else
// cuFFT expects C++ name mangling, so the linkage must be C++.
#define CALLBACK_LINKAGE
#include <cuComplex.h>
typedef cuDoubleComplex hipDoubleComplex;
#define make_hipDoubleComplex make_cuDoubleComplex 
#define hipCmul cuCmul
#endif

struct load_cbdata
{
    hipDoubleComplex* filter;
    double            scale;
};

/// \brief \return the \p input multiplied with both the filter and the scale
/// from the \p cbdata.
CALLBACK_LINKAGE
__device__ hipDoubleComplex load_callback(void*              input_void,
                                          unsigned long long offset,
                                          void*              cbdata,
                                          void* /*sharedMem*/)
{
    auto input = static_cast<hipDoubleComplex*>(input_void);
    auto data = static_cast<load_cbdata*>(cbdata);

    return hipCmul(hipCmul(input[offset], data->filter[offset]),
                   make_hipDoubleComplex(data->scale, 0));
}
)";

int main()
{
    std::cout << "hipfft 1D double-precision complex-to-complex transform with callback\n";

    constexpr int Nx        = 8; // Size of data vector
    constexpr int direction = HIPFFT_FORWARD; // forward=-1, backward=1

    // Initialize data and filter on host
    std::vector<hipfftDoubleComplex>       h_data(Nx), h_filter(Nx);
    std::random_device                     rd;
    std::default_random_engine             gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for(size_t i = 0; i < Nx; i++)
    {
        h_data[i].x   = i;
        h_data[i].y   = i;
        h_filter[i].x = distribution(gen);
    }

    const size_t complex_bytes = sizeof(decltype(h_data)::value_type) * h_data.size();

    // Create HIP device object and copy data to device
    hipfftDoubleComplex *d_data, *d_filter;
    HIP_CHECK(hipMalloc(&d_data, complex_bytes));
    HIP_CHECK(hipMalloc(&d_filter, complex_bytes));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), complex_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_filter, h_filter.data(), complex_bytes, hipMemcpyHostToDevice));

    std::cout << "input:\n";
    for(size_t i = 0; i < h_data.size(); i++)
    {
        std::cout << "(" << h_data[i].x << ", " << h_data[i].y << ") ";
    }
    std::cout << std::endl;

    // Prepare callback
    load_cbdata h_callback_data;
    h_callback_data.filter = d_filter;
    h_callback_data.scale  = 1.0 / static_cast<double>(Nx);
    void* d_callback_data;
    HIP_CHECK(hipMalloc(&d_callback_data, sizeof(load_cbdata)));
    HIP_CHECK(
        hipMemcpy(d_callback_data, &h_callback_data, sizeof(load_cbdata), hipMemcpyHostToDevice));

    auto load_callback_code = compile_jit_callback(load_callback_src);

    std::vector<void*> cbdatas(1);
    cbdatas[0] = d_callback_data;

    // Create the plan
    hipfftHandle plan;
    HIPFFT_CHECK(hipfftCreate(&plan));

    // Set callback
    HIPFFT_CHECK(hipfftXtSetJITCallback(plan,
                                        "load_callback",
                                        load_callback_code.data(),
                                        load_callback_code.size(),
                                        HIPFFT_CB_LD_COMPLEX_DOUBLE,
                                        cbdatas.data()));

    size_t workSize = 0;
    HIPFFT_CHECK(hipfftMakePlan1d(plan, // Plan handle
                                  Nx, // Transform length
                                  HIPFFT_Z2Z, // Transform type
                                  1, // Number of transforms
                                  &workSize)); // Work memory

    // Execute plan
    HIPFFT_CHECK(hipfftExecZ2Z(plan, d_data, d_data, direction));

    std::cout << "output:\n";
    HIP_CHECK(hipMemcpy(h_data.data(), d_data, complex_bytes, hipMemcpyDeviceToHost));
    for(size_t i = 0; i < h_data.size(); i++)
    {
        std::cout << "(" << h_data[i].x << ", " << h_data[i].y << ") ";
    }
    std::cout << std::endl;

    // Clean up
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(d_callback_data));
    HIP_CHECK(hipFree(d_filter));
    HIP_CHECK(hipFree(d_data));

    return 0;
}
