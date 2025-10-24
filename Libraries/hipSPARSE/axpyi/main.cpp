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
#include "hipsparse_utils.hpp"

#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Number of non-zeros of the sparse vector
    int nnz = 3;

    // Sparse index vector
    int hx_ind[3] = {0, 3, 5};

    // Sparse value vector
    double hx_val[3] = {1.0, 2.0, 3.0};

    // Dense vector
    double hy[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Scalar alpha
    double alpha = 3.7;

    // Index base
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    // Offload data to device
    int*    dx_ind;
    double* dx_val;
    double* dy;

    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * 9));

    HIP_CHECK(hipMemcpy(dx_ind, hx_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_val, hx_val, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(double) * 9, hipMemcpyHostToDevice));

    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Call daxpyi to perform y = y + alpha * x
    HIPSPARSE_CHECK(hipsparseDaxpyi(handle, nnz, &alpha, dx_val, dx_ind, dy, idx_base));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * 9, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));

    return 0;
}
