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

#include "example_utils.hpp"
#include "rocsparse_utils.hpp"

#include <rocsparse/rocsparse.h>

#include <hip/hip_runtime.h>

#include <array>
#include <iostream>

int main()
{
    // Number of elements in dense vector
    constexpr rocsparse_int size = 9;

    // Number of non-zeros of the sparse vector
    constexpr rocsparse_int nnz = 5;

    // Sparse index vector
    constexpr std::array<rocsparse_int, nnz> hx_ind = {0, 1, 3, 5, 8};

    // Sparse value vector
    constexpr std::array<float, nnz> hx_val = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Dense vector
    constexpr std::array<float, size> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // Index base
    constexpr rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    
    // Offload data to device
    rocsparse_int* dx_ind;
    float*        dx_val;
    float*        dy;

    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(dx_ind, hx_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_val, hx_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice));

    // Call sdoti to compute the dot product
    float dot;
    ROCSPARSE_CHECK(rocsparse_sdoti(handle, nnz, dx_val, dx_ind, dy, &dot, idx_base));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));

    // 8. Print results to standard output.
    std::cout << "Solution successfully computed: ";

    std::cout << "dot = " << dot << std::endl;

    float host_dot = 0.0f;
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        host_dot += hx_val[i] * hy[hx_ind[i] - idx_base];
    }

    const float eps = 1.0e5 * std::numeric_limits<float>::epsilon();
    int errors = std::fabs(host_dot - dot) > eps;

    // Print validation result.
    return report_validation_result(errors);
}