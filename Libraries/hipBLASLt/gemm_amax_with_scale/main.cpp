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

#include <hipblaslt/hipblaslt-ext-op.h>

void gemm_amax_with_scale(hipblasLtHandle_t  handle,
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
    /** This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, float>
        runner(1024, 512, 1024, 1, 1.f, 0.f, 32 * 1024 * 1024);

    runner.run(
        [&runner]
        {
            gemm_amax_with_scale(runner.handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
                                 runner.m,
                                 runner.n,
                                 runner.k,
                                 runner.batch_count,
                                 runner.alpha,
                                 runner.beta,
                                 runner.d_a,
                                 runner.d_b,
                                 runner.d_c,
                                 runner.d_d,
                                 runner.d_workspace,
                                 runner.max_workspace_size,
                                 runner.stream);
        });

    return 0;
}

void gemm_amax_with_scale(hipblasLtHandle_t  handle,
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
    (void)batch_count;

    // allocate data for amax
    void *in_scale, *out_amax; // host
    void *d_in_scale, *d_out_amax; // device

    HIP_CHECK(hipMalloc(&d_in_scale, 1 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_amax, 1 * sizeof(float)));

    HIP_CHECK(hipHostMalloc(&in_scale, 1 * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&out_amax, 1 * sizeof(float)));

    // copy amax data to device
    *(float*)in_scale = (float)0.5;
    HIP_CHECK(
        hipMemcpyAsync(d_in_scale, in_scale, 1 * sizeof(float), hipMemcpyHostToDevice, stream));

    // set matrix layout for gemm
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_8F_E4M3_FNUZ, m, k, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_8F_E4M3_FNUZ, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_8F_E4M3_FNUZ, m, n, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_8F_E4M3_FNUZ, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a,
                                                    sizeof(int32_t)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b,
                                                    sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue,
                                                    sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                    &d_out_amax,
                                                    sizeof(void*)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &d_in_scale,
                                                    sizeof(void*)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &max_workspace_size,
                                                          sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristic_result[request_solutions];
    int                              returned_algo_count = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    matmul,
                                                    mat_a,
                                                    mat_b,
                                                    mat_c,
                                                    mat_d,
                                                    pref,
                                                    request_solutions,
                                                    heuristic_result,
                                                    &returned_algo_count));

    if(returned_algo_count == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        HIP_CHECK(hipFree(d_in_scale));
        HIP_CHECK(hipFree(d_out_amax));
        HIP_CHECK(hipFree(in_scale));
        HIP_CHECK(hipFree(out_amax));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
        HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
        return;
    }

    uint64_t workspace_size = max_workspace_size;
    for(int i = 0; i < returned_algo_count; i++)
        workspace_size = std::max(workspace_size, heuristic_result[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // HIP_CHECKhipMalloc(&d_workspace, workspace_size));

    HIPBLASLT_CHECK(hipblasLtMatmul(handle,
                                    matmul,
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
                                    workspace_size,
                                    stream));

    // deallocate memory space of amax
    HIP_CHECK(hipFree(d_in_scale));
    HIP_CHECK(hipFree(d_out_amax));
    HIP_CHECK(hipFree(in_scale));
    HIP_CHECK(hipFree(out_amax));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
