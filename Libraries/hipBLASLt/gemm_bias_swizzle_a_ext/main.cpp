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

#include <cstring>

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
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8F_E4M3_FNUZ:
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
    std::memcpy(tmp_tensor.template as<T>(), src, m * k * sizeof(T));

    if(col_maj)
    {
        auto org_tensor = tensor::create<T>({k, m});
        std::memcpy(org_tensor.template as<T>(), src, m * k * sizeof(T));
        tmp_tensor = permute_tensor<T>(org_tensor, {1, 0});
    }
    auto       multiple_m = mi_m;
    auto       multiple_k = mi_k * pack_k;
    const auto padded_m   = (m / multiple_m + !!(m % multiple_m)) * multiple_m;
    const auto padded_k   = (k / multiple_k + !!(k % multiple_k)) * multiple_k;
    shape_t    padded_shape{padded_m, padded_k};
    auto       padded_tensor = pad_tensor<T>(tmp_tensor, padded_shape, T(0));
    padded_tensor.reshape(
        {padded_m / mi_m, mi_m, padded_k / (mi_k * pack_k), mi_k / mi_kv, mi_kv * pack_k});
    tensor permuted = permute_tensor<T>(padded_tensor, {0, 2, 3, 1, 4});
    std::memcpy(dst, permuted.template as<T>(), padded_m * padded_k * sizeof(T));
}

void swizzle_gemm_epilogue_bias_vec_ext(hipblasLtHandle_t  handle,
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
                                        bool               swizzle_a,
                                        hipStream_t        stream);

int main()
{
    constexpr int64_t m{5280};
    constexpr int64_t n{2048};
    constexpr int64_t k{1024};

    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        swizzle_runner_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzle_runner_inst.run(
        [&swizzle_runner_inst]
        {
            swizzle_gemm_epilogue_bias_vec_ext(swizzle_runner_inst.handle,
                                               /*For swizzle-A, it forces to use TN*/
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
                                               true,
                                               swizzle_runner_inst.stream);
        });

    return 0;
}

void swizzle_gemm_epilogue_bias_vec_ext(hipblasLtHandle_t  handle,
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
                                        bool               swizzle_a,
                                        hipStream_t        stream)
{
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_16F, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16F, m, n, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16F, m, n, m));

    if(trans_a == HIPBLAS_OP_T)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_16F, k, m, k));

        if(swizzle_a)
        {
            hipblasLtOrder_t order_a = HIPBLASLT_ORDER_COL16_4R8;
            HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_a,
                                                              HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                              &order_a,
                                                              sizeof(order_a)));
            std::vector<hipblasLtHalf> src(m * k, 0);
            std::vector<hipblasLtHalf> dst(m * k, 0);
            HIP_CHECK(
                hipMemcpy(src.data(), d_a, m * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
            swizzle_tensor(dst.data(), src.data(), m, k, true);
            HIP_CHECK(
                hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
        }
    }
    else
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_16F, m, k, m));
    }

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

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue,
                                                    sizeof(epilogue)));
    // Allocate and set the bias tensor
    std::vector<hipblasLtHalf> h_bias(m, 1.0f); // Example bias values, adjust as needed
    void*                      d_bias;
    HIP_CHECK(hipMalloc(&d_bias, m * sizeof(hipblasLtHalf))); // Allocate memory for bias
    HIP_CHECK(hipMemcpy(d_bias,
                        h_bias.data(),
                        m * sizeof(hipblasLtHalf),
                        hipMemcpyHostToDevice)); // Copy bias to device

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &d_bias,
                                                    sizeof(void*)));

    hipblaslt_ext::Gemm
        gemm(handle, matmul, &alpha, d_a, mat_a, d_b, mat_b, &beta, d_c, mat_c, d_d, mat_d);

    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(max_workspace_size);

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results;
    const int                                     requested_solutions = 1;
    HIPBLASLT_CHECK(gemm.algoGetHeuristic(requested_solutions, gemm_pref, heuristic_results));

    if(heuristic_results.size() == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
        HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
        HIP_CHECK(hipFree(d_bias));
        return;
    }

    HIPBLASLT_CHECK(gemm.initialize(heuristic_results[0].algo, d_workspace));
    HIPBLASLT_CHECK(gemm.run(stream));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
    HIP_CHECK(hipFree(d_bias));
    return;
}
