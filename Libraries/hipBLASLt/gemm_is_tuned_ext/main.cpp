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
#include "hipblaslt_utils.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>

void print_result(int tuned, uint64_t m, uint64_t n, uint64_t k)
{
    if(tuned == 1)
    {
        std::cout << "[" << m << ", " << n << ", " << k << "] is tuned\n";
    }
    else
    {
        std::cout << "[" << m << ", " << n << ", " << k << "] is un-tuned\n";
    }
}

int main(int argc, char** argv)
{
    hipblasLtHandle_t handle{};
    hipblasLtCreate(&handle);
    hipblasLtMatmulDesc_t   matmul_desc{};
    hipblasLtMatrixLayout_t mat_a{};
    hipblasLtMatrixLayout_t mat_b{};
    hipblasLtMatrixLayout_t mat_c{};
    hipblasLtMatrixLayout_t mat_d{};
    hipblasLtMatmulDescCreate(&matmul_desc, hipblasComputeType_t::HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t op_a = HIPBLAS_OP_T;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    hipblasLtPointerMode_t p_mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    hipblasLtMatmulDescSetAttribute(matmul_desc,
                                    HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                    &p_mode,
                                    sizeof(p_mode));
    const uint64_t m = argc > 3 ? std::atoll(argv[1]) : 128;
    const uint64_t n = argc > 3 ? std::atoll(argv[2]) : 128;
    const uint64_t k = argc > 3 ? std::atoll(argv[3]) : 128;
    hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_16F, k, m, k);
    hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_16F, k, n, k);
    hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16F, m, n, m);
    hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16F, m, n, m);
    auto tuned = hipblaslt_ext::matmulIsTuned(handle, matmul_desc, mat_a, mat_b, mat_c, mat_d);
    print_result(tuned, m, n, k);
    hipblasLtMatmulDescDestroy(matmul_desc);
    hipblasLtMatrixLayoutDestroy(mat_a);
    hipblasLtMatrixLayoutDestroy(mat_b);
    hipblasLtMatrixLayoutDestroy(mat_c);
    hipblasLtMatrixLayoutDestroy(mat_d);
    hipblasLtDestroy(handle);
    return 0;
}
