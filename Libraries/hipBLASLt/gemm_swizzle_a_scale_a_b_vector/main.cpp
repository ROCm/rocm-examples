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

void calculate_k_for_swizzling(hipDataType datatype, size_t& mi_k, size_t& mi_kv, size_t& pack_k)
{
    switch(datatype)
    {
        case HIP_R_32F:
            mi_k  = 4;
            mi_kv = 1;
            break;
        case HIP_R_16F:
        case HIP_R_16BF:
            mi_k  = 16;
            mi_kv = 4;
            break;
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
            mi_k  = 32;
            mi_kv = 8;
            break;
        default: std::cerr << "unsupported datatype in calculate_k_for_swizzling" << '\n';
    }

    pack_k = 16 / mi_kv / real_datatype_size(datatype);
}

template<typename T>
void swizzle_tensor(T* dst, const T* src, size_t m, size_t k, bool col_maj)
{
    using namespace tensor_manipulation;
    size_t mi_m = 16;
    size_t mi_k = 0, mi_kv = 0, pack_k = 0;
    calculate_k_for_swizzling(hipblaslt_type_to_datatype<T>(), mi_k, mi_kv, pack_k);
    auto tmp_tensor = tensor::create<T>({m, k});
    std::copy(src, src + (m * k), tmp_tensor.template as<T>());

    if(col_maj)
    {
        auto org_tensor = tensor::create<T>({k, m});
        std::copy(src, src + (m * k), org_tensor.template as<T>());
        tmp_tensor = permute_tensor<T>(org_tensor, {1, 0});
    }

    tmp_tensor.reshape({m / mi_m, mi_m, k / (mi_k * pack_k), mi_k / mi_kv, mi_kv * pack_k});
    tensor permuted = permute_tensor<T>(tmp_tensor, {0, 2, 3, 1, 4});
    std::copy(permuted.template as<T>(), permuted.template as<T>() + (m * k), dst);
}

void simple_gemm(hipblasLtHandle_t  handle,
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
                 hipDataType        ti_ab,
                 bool               swizzle_a,
                 hipStream_t        stream,
                 const float*       h_scale_a_vec,
                 const float*       h_scale_b_vec);

int main()
{
    constexpr int64_t m{5280};
    constexpr int64_t n{2048};
    constexpr int64_t k{1024};

    // Non-swizzle runner: TN, ScaleABVec, batch count = 1, alpha, beta = 1.0f
    runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hip_bfloat16, float, float>
        runner_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    std::vector<float> scale_a_vec = std::vector<float>(m, 0.5f); // scale A vector = vector len M
    std::vector<float> scale_b_vec = std::vector<float>(n, 2.0f); // scale B vector = vector len N
    std::cout << "Running with Scale A Vector with all values = " << scale_a_vec[0]
              << " and Scale B Vector with all values = " << scale_b_vec[0] << std::endl;

    runner_inst.run(
        [&runner_inst, scale_a_vec, scale_b_vec]
        {
            simple_gemm(runner_inst.handle,
                        HIPBLAS_OP_T,
                        HIPBLAS_OP_N,
                        runner_inst.m,
                        runner_inst.n,
                        runner_inst.k,
                        runner_inst.batch_count,
                        runner_inst.alpha,
                        runner_inst.beta,
                        runner_inst.d_a,
                        runner_inst.d_b,
                        runner_inst.d_c,
                        runner_inst.d_d,
                        runner_inst.d_workspace,
                        runner_inst.max_workspace_size,
                        HIP_R_8F_E4M3_FNUZ,
                        false,
                        runner_inst.stream,
                        scale_a_vec.data(),
                        scale_b_vec.data());
        });

    // swizzleA runner: TN, ScaleABVec, batch count = 1, alpha, beta = 1.0f
    runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hip_bfloat16, float, float>
        swizzle_runner_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzle_runner_inst.run(
        [&swizzle_runner_inst, &runner_inst, scale_a_vec, scale_b_vec]
        {
            // copy inputs from first runner for comparison and validation
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_a,
                                runner_inst.d_a,
                                m * k * sizeof(hipblaslt_f8_fnuz),
                                hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_b,
                                runner_inst.d_b,
                                n * k * sizeof(hipblaslt_f8_fnuz),
                                hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_c,
                                runner_inst.d_c,
                                m * n * sizeof(hip_bfloat16),
                                hipMemcpyDeviceToDevice));

            simple_gemm(swizzle_runner_inst.handle,
                        HIPBLAS_OP_T,
                        HIPBLAS_OP_N,
                        swizzle_runner_inst.m,
                        swizzle_runner_inst.n,
                        swizzle_runner_inst.k,
                        swizzle_runner_inst.batch_count,
                        swizzle_runner_inst.alpha,
                        swizzle_runner_inst.beta,
                        swizzle_runner_inst.d_a,
                        swizzle_runner_inst.d_b,
                        swizzle_runner_inst.d_c,
                        swizzle_runner_inst.d_d,
                        swizzle_runner_inst.d_workspace,
                        swizzle_runner_inst.max_workspace_size,
                        HIP_R_8F_E4M3_FNUZ,
                        true,
                        swizzle_runner_inst.stream,
                        scale_a_vec.data(),
                        scale_b_vec.data());
        });

    // Compare results from non-swizzling with swizzling
    const hip_bfloat16* regular_cpu_d  = static_cast<hip_bfloat16*>(runner_inst.d);
    const hip_bfloat16* swizzled_cpu_d = static_cast<hip_bfloat16*>(swizzle_runner_inst.d);

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regular_cpu_d[i] - float(swizzled_cpu_d[i])));
        if(diff > 1e-5)
        {
            std::cerr << "F8 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
            break;
        }
    }

    std::cout << "Matrix multiplication and validation completed successfully." << std::endl;

    return 0;
}

void simple_gemm(hipblasLtHandle_t  handle,
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
                 hipDataType        ti_ab,
                 bool               swizzle_a,
                 hipStream_t        stream,
                 const float*       h_scale_a_vec,
                 const float*       h_scale_b_vec)
{
    (void)batch_count;

    // Scale A, B Vector
    float* d_scale_a_vec;
    float* d_scale_b_vec;
    HIP_CHECK(hipMalloc(&d_scale_a_vec, m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_scale_b_vec, n * sizeof(float)));
    HIP_CHECK(hipMemcpyAsync(d_scale_a_vec,
                             h_scale_a_vec,
                             m * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(d_scale_b_vec,
                             h_scale_b_vec,
                             n * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));

    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, ti_ab, k, m, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, ti_ab, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16BF, m, n, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16BF, m, n, m));

    // swizzle case and input = FP8
    if(swizzle_a && ti_ab == HIP_R_8F_E4M3_FNUZ)
    {
        hipblasLtOrder_t order_a = HIPBLASLT_ORDER_COL16_4R16;
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_a,
                                                          HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                          &order_a,
                                                          sizeof(order_a)));
        std::vector<hipblaslt_f8_fnuz> src(m * k, 0);
        std::vector<hipblaslt_f8_fnuz> dst(m * k, 0);

        // pre-shuffle input data in host memory
        HIP_CHECK(
            hipMemcpy(src.data(), d_a, m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyDeviceToHost));
        swizzle_tensor(dst.data(), src.data(), m, k, false);
        HIP_CHECK(
            hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyHostToDevice));
    }

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

    // Set ScaleA, B mode (Vector)
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                    &mode,
                                                    sizeof(uint32_t)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &mode,
                                                    sizeof(uint32_t)));

    // Set A and B matrix scale factors
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &d_scale_a_vec,
                                                    sizeof(float*)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &d_scale_b_vec,
                                                    sizeof(float*)));

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
        HIP_CHECK(hipFree(d_scale_a_vec));
        HIP_CHECK(hipFree(d_scale_b_vec));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
        HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returned_algo_count; i++)
        workspace_size = std::max(workspace_size, heuristic_result[i].workspaceSize);

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

    // Clean up resources
    HIP_CHECK(hipFree(d_scale_a_vec));
    HIP_CHECK(hipFree(d_scale_b_vec));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
