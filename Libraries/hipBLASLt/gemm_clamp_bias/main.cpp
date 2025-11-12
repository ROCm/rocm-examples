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

#include <hipblaslt/hipblaslt.h>
#include <iostream>

void simple_gemm_clamp_bias(hipblasLtHandle_t  handle,
                            hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b,
                            int64_t            m,
                            int64_t            n,
                            int64_t            k,
                            int64_t            batch_count,
                            float&             alpha,
                            float&             beta,
                            void*              d_a,
                            void*              d_b,
                            void*              d_c,
                            void*              d_d,
                            void*              d_workspace,
                            int64_t            max_workspace_size,
                            hipStream_t        stream);

int main()
{
    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        simple_runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    simple_runner.run(
        [&simple_runner]
        {
            simple_gemm_clamp_bias(simple_runner.handle,
                                   HIPBLAS_OP_N,
                                   HIPBLAS_OP_N,
                                   simple_runner.m,
                                   simple_runner.n,
                                   simple_runner.k,
                                   simple_runner.batch_count,
                                   simple_runner.alpha,
                                   simple_runner.beta,
                                   simple_runner.d_a,
                                   simple_runner.d_b,
                                   simple_runner.d_c,
                                   simple_runner.d_d,
                                   simple_runner.d_workspace,
                                   simple_runner.max_workspace_size,
                                   simple_runner.stream);
        });

    return 0;
}

void simple_gemm_clamp_bias(hipblasLtHandle_t  handle,
                            hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b,
                            int64_t            m,
                            int64_t            n,
                            int64_t            k,
                            int64_t            batch_count,
                            float&             alpha,
                            float&             beta,
                            void*              d_a,
                            void*              d_b,
                            void*              d_c,
                            void*              d_d,
                            void*              d_workspace,
                            int64_t            max_workspace_size,
                            hipStream_t        stream)
{
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_16F, m, k, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_16F, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16F, m, n, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16F, m, n, m));

    if(batch_count > 1)
    {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_a,
                                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count,
                                                          sizeof(batch_count)));
        HIPBLASLT_CHECK(
            hipblasLtMatrixLayoutSetAttribute(mat_a,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_a,
                                              sizeof(stride_a)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_b,
                                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count,
                                                          sizeof(batch_count)));
        HIPBLASLT_CHECK(
            hipblasLtMatrixLayoutSetAttribute(mat_b,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_b,
                                              sizeof(stride_b)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_c,
                                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count,
                                                          sizeof(batch_count)));
        HIPBLASLT_CHECK(
            hipblasLtMatrixLayoutSetAttribute(mat_c,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_c,
                                              sizeof(stride_c)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_d,
                                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count,
                                                          sizeof(batch_count)));
        HIPBLASLT_CHECK(
            hipblasLtMatrixLayoutSetAttribute(mat_d,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_d,
                                              sizeof(stride_d)));
    }

    hipblasLtMatmulDesc_t mat_mul;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&mat_mul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a,
                                                    sizeof(int32_t)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b,
                                                    sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue,
                                                    sizeof(epilogue)));
    float clamp_lower = -1.5f, clamp_upper = 1.5f;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
                                                    &clamp_lower,
                                                    sizeof(float)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
                                                    &clamp_upper,
                                                    sizeof(float)));

    // Set Desc Bias Data Type
    int32_t bias_type = HIP_R_16F;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &bias_type,
                                                    sizeof(bias_type)));

    // Allocate and set the bias tensor
    std::vector<hipblasLtHalf> h_bias(
        m,
        static_cast<hipblasLtHalf>(1.0)); // Example bias values, adjust as needed
    void* d_bias;
    HIP_CHECK(hipMalloc(&d_bias, m * sizeof(hipblasLtHalf)));
    HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), m * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(mat_mul,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &d_bias,
                                                    sizeof(void*)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &max_workspace_size,
                                                          sizeof(max_workspace_size)));

    const int                        requested_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristic_result[requested_solutions];
    int                              returned_algo_count = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    mat_mul,
                                                    mat_a,
                                                    mat_b,
                                                    mat_c,
                                                    mat_d,
                                                    preference,
                                                    requested_solutions,
                                                    heuristic_result,
                                                    &returned_algo_count));

    if(returned_algo_count == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
    }
    else
    {
        uint64_t workspace_bytes = 0;
        for(int i = 0; i < returned_algo_count; i++)
        {
            workspace_bytes = max(workspace_bytes, heuristic_result[i].workspaceSize);
        }
        // In this sample, the workspace is already allocated with max_workspace_size
        // If not, allocate d_workspace here
        // HIP_CHECK(hipMalloc(&d_workspace, workspace_bytes));

        HIPBLASLT_CHECK(hipblasLtMatmul(handle,
                                        mat_mul,
                                        &alpha,
                                        d_a,
                                        mat_a,
                                        d_b,
                                        mat_b,
                                        &beta,
                                        d_c,
                                        mat_c,
                                        d_d,
                                        mat_d,
                                        &heuristic_result[0].algo,
                                        d_workspace,
                                        workspace_bytes,
                                        stream));
    }
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(mat_mul));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(preference));
    HIP_CHECK(hipFree(d_bias));
    return;
}
