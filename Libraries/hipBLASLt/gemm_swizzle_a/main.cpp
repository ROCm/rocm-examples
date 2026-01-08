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

void calculate_k_for_swizzling(hipDataType datatype, size_t& mik, size_t& mikv, size_t& pack_k)
{
    switch(datatype)
    {
        case HIP_R_32F:
            mik  = 4;
            mikv = 1;
            break;
        case HIP_R_16F:
        case HIP_R_16BF:
            mik  = 16;
            mikv = 4;
            break;
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
            mik  = 32;
            mikv = 8;
            break;
        default: std::cerr << "unsupported datatype in calculate_k_for_swizzling" << '\n';
    }

    pack_k = 16 / mikv / real_datatype_size(datatype);
}

template<typename T>
void swizzle_tensor(T* dst, const T* src, size_t m, size_t k, bool col_maj)
{
    using namespace tensor_manipulation;
    size_t mim = 16;
    size_t mik = 0, mikv = 0, pack_k = 0;
    calculate_k_for_swizzling(hipblaslt_type_to_datatype<T>(), mik, mikv, pack_k);
    auto tmp_tensor = tensor::create<T>({m, k});
    std::copy(src, src + m * k, tmp_tensor.template as<T>());

    if(col_maj)
    {
        auto org_tensor = tensor::create<T>({k, m});
        std::copy(src, src + m * k, org_tensor.template as<T>());
        tmp_tensor = permute_tensor<T>(org_tensor, {1, 0});
    }

    tmp_tensor.reshape({m / mim, mim, k / (mik * pack_k), mik / mikv, mikv * pack_k});
    tensor permuted = permute_tensor<T>(tmp_tensor, {0, 2, 3, 1, 4});
    std::copy(permuted.template as<T>(), permuted.template as<T>() + m * k, dst);
}

void gemm(hipblasLtHandle_t  handle,
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
          hipDataType        TiAB,
          bool               swizzle_a,
          hipStream_t        stream);

int main()
{
    constexpr int64_t m{5280};
    constexpr int64_t n{2048};
    constexpr int64_t k{1024};
    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        runner_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    runner_inst.run(
        [&runner_inst]
        {
            gemm(runner_inst.handle,
                 HIPBLAS_OP_N,
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
                 HIP_R_16F,
                 false,
                 runner_inst.stream);
        });

    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        swizzle_runner_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzle_runner_inst.run(
        [&swizzle_runner_inst, &runner_inst]
        {
            // copy inputs from first runner for comparison and validation
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_a,
                                runner_inst.d_a,
                                m * k * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_b,
                                runner_inst.d_b,
                                n * k * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_inst.d_c,
                                runner_inst.d_c,
                                m * n * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToDevice));
            /** This is an example with swizzle-A
         *  a = (k, m). lda = k
         *  b = (k, n). ldb = k
         *  c = d = (m, n). ldc = ldd = m
         */
            gemm(swizzle_runner_inst.handle,
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
                 HIP_R_16F,
                 true,
                 swizzle_runner_inst.stream);
        });

    runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, float>
        swizzle_runner_f8_inst(m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzle_runner_f8_inst.run(
        [&swizzle_runner_f8_inst, &runner_inst]
        {
            // convert inputs from reference runner to fp8
            std::vector<hipblasLtHalf>     cup_a_f16(m * k, hipblasLtHalf(0.f));
            std::vector<hipblasLtHalf>     cup_b_f16(k * n, hipblasLtHalf(0.f));
            std::vector<hipblaslt_f8_fnuz> cup_a_f8(m * k, hipblaslt_f8_fnuz(0.f));
            std::vector<hipblaslt_f8_fnuz> cup_b_f8(k * n, hipblaslt_f8_fnuz(0.f));

            HIP_CHECK(hipMemcpy(cup_a_f16.data(),
                                runner_inst.d_a,
                                cup_a_f16.size() * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(cup_b_f16.data(),
                                runner_inst.d_b,
                                cup_b_f16.size() * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToHost));

            for(size_t i = 0; i < cup_a_f16.size(); ++i)
            {
                cup_a_f8[i] = hipblaslt_f8_fnuz(float(cup_a_f16[i]));
            }

            for(size_t i = 0; i < cup_b_f16.size(); ++i)
            {
                cup_b_f8[i] = hipblaslt_f8_fnuz(float(cup_b_f16[i]));
            }

            // copy inputs from first runner for comparison and validation
            HIP_CHECK(hipMemcpy(swizzle_runner_f8_inst.d_a,
                                cup_a_f8.data(),
                                m * k * sizeof(hipblaslt_f8_fnuz),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_f8_inst.d_b,
                                cup_b_f8.data(),
                                n * k * sizeof(hipblaslt_f8_fnuz),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(swizzle_runner_f8_inst.d_c,
                                runner_inst.d_c,
                                m * n * sizeof(hipblasLtHalf),
                                hipMemcpyDeviceToDevice));
            /** This is an example with swizzle-A
         *  a = (k, m). lda = k
         *  b = (k, n). ldb = k
         *  c = d = (m, n). ldc = ldd = m
         */
            gemm(swizzle_runner_f8_inst.handle,
                 /*For swizzle-A, it forces to use TN*/
                 HIPBLAS_OP_T,
                 HIPBLAS_OP_N,
                 swizzle_runner_f8_inst.m,
                 swizzle_runner_f8_inst.n,
                 swizzle_runner_f8_inst.k,
                 swizzle_runner_f8_inst.batch_count,
                 swizzle_runner_f8_inst.alpha,
                 swizzle_runner_f8_inst.beta,
                 swizzle_runner_f8_inst.d_a,
                 swizzle_runner_f8_inst.d_b,
                 swizzle_runner_f8_inst.d_c,
                 swizzle_runner_f8_inst.d_d,
                 swizzle_runner_f8_inst.d_workspace,
                 swizzle_runner_f8_inst.max_workspace_size,
                 HIP_R_8F_E4M3_FNUZ,
                 true,
                 swizzle_runner_f8_inst.stream);
        });

    const hipblasLtHalf* regular_cpu_d     = static_cast<hipblasLtHalf*>(runner_inst.d);
    const hipblasLtHalf* swizzled_cpu_d    = static_cast<hipblasLtHalf*>(swizzle_runner_inst.d);
    const hipblasLtHalf* swizzled_cpu_d_f8 = static_cast<hipblasLtHalf*>(swizzle_runner_f8_inst.d);

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regular_cpu_d[i] - float(swizzled_cpu_d[i])));
        if(diff > 1e-5)
        {
            std::cerr << "F16 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
            break;
        }
    }

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regular_cpu_d[i] - float(swizzled_cpu_d_f8[i])));
        if(diff > 1e-5)
        {
            std::cerr << "f8 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
            break;
        }
    }

    return 0;
}

void gemm(hipblasLtHandle_t  handle,
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
          hipDataType        TiAB,
          bool               swizzle_a,
          hipStream_t        stream)
{
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, TiAB, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16F, m, n, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16F, m, n, m));

    if(trans_a == HIPBLAS_OP_T)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, TiAB, k, m, k));

        if(swizzle_a && TiAB == HIP_R_16F)
        {
            hipblasLtOrder_t orderA = HIPBLASLT_ORDER_COL16_4R8;
            HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_a,
                                                              HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                              &orderA,
                                                              sizeof(orderA)));
            std::vector<hipblasLtHalf> src(m * k, 0);
            std::vector<hipblasLtHalf> dst(m * k, 0);
            HIP_CHECK(
                hipMemcpy(src.data(), d_a, m * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
            swizzle_tensor(dst.data(), src.data(), m, k, true);
            HIP_CHECK(
                hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
        }
        else if(swizzle_a && TiAB == HIP_R_8F_E4M3_FNUZ)
        {
            hipblasLtOrder_t orderA = HIPBLASLT_ORDER_COL16_4R16;
            HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(mat_a,
                                                              HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                              &orderA,
                                                              sizeof(orderA)));
            std::vector<hipblaslt_f8_fnuz> src(m * k, 0);
            std::vector<hipblaslt_f8_fnuz> dst(m * k, 0);
            HIP_CHECK(
                hipMemcpy(src.data(), d_a, m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyDeviceToHost));
            swizzle_tensor(dst.data(), src.data(), m, k, true);
            HIP_CHECK(
                hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyHostToDevice));
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

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue,
                                                    sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &max_workspace_size,
                                                          sizeof(max_workspace_size)));

    const int                        request_solutions = 100;
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
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // HIP_CHECKhipMalloc(&d_workspace, workspace_size));
    float         best_time_ms = std::numeric_limits<float>::max();
    constexpr int num_warmup_runs{100};
    constexpr int num_runs{1000};

    for(int j = 0; j < returned_algo_count; ++j)
    {
        for(int i = 0; i < num_warmup_runs; ++i)
        {
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
                                            &heuristic_result[j].algo,
                                            d_workspace,
                                            workspace_size,
                                            stream));
        }

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, stream));

        for(int i = 0; i < num_runs; ++i)
        {
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
                                            &heuristic_result[j].algo,
                                            d_workspace,
                                            workspace_size,
                                            stream));
        }

        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipDeviceSynchronize());
        float timeMs{};
        HIP_CHECK(hipEventElapsedTime(&timeMs, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        best_time_ms = std::min(timeMs, best_time_ms);
    }

    std::cout << "Best solution time: " << best_time_ms / num_runs * 1000
              << " us (swizzle_a == " << int(swizzle_a) << ")\n";
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
