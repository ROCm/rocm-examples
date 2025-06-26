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
    constexpr rocsparse_int nnz = 3;

    // Sparse index vector
    constexpr std::array<rocsparse_int, nnz> hx_ind = {0, 1, 8};

    // Sparse value vector
    std::array<double, nnz> hx_val = {4.0, 5.0, 10.0};

    // Dense vector
    std::array<double, size> hy = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    std::array<double, nnz> hx_val_host(hx_val);
    std::array<double, size> hy_host(hy);

    // c and s
    double c = 3.0;
    double s = 2.0;

    // Index base
    rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    
    // Offload data to device
    rocsparse_int* dx_ind;
    double*        dx_val;
    double*        dy;

    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * size));

    HIP_CHECK(hipMemcpy(dx_ind, hx_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_val, hx_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(double) * size, hipMemcpyHostToDevice));

    // Call sroti
    ROCSPARSE_CHECK(rocsparse_droti(handle, nnz, dx_val, dx_ind, dy, &c, &s, idx_base));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hx_val.data(), dx_val, sizeof(double) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(double) * size, hipMemcpyDeviceToHost));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));

    // 8. Print results to standard output.
    std::cout << "Solution successfully computed: ";

    std::cout << "x_val = " << format_range(std::begin(hx_val), std::end(hx_val)) << std::endl;
    std::cout << "y = " << format_range(std::begin(hy), std::end(hy)) << std::endl;

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        double x_tmp = hx_val_host[i];
        double y_tmp = hy_host[hx_ind[i] - idx_base];

        hx_val_host[i]    = c * x_tmp + s * y_tmp;
        hy_host[hx_ind[i] - idx_base] = c * y_tmp - s * x_tmp;
    }

    int          errors{};
    for(size_t i = 0; i < hx_val.size(); ++i)
    {
        errors += std::abs(hx_val[i] - hx_val_host[i]);
    }
    for(size_t i = 0; i < hy.size(); ++i)
    {
        errors += std::abs(hy[i] - hy_host[i]);
    }

    // Print validation result.
    return report_validation_result(errors);
}
