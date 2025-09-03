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

void gemm_mix_precision_ext(hipblasLtHandle_t  handle,
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

template<typename TypeA, typename TypeB, typename TypeCD, typename AlphaType, typename BetaType>
int validate(const runner<TypeA, TypeB, TypeCD, AlphaType, BetaType>& runner)
{
    std::vector<float> ref(runner.m * runner.n * runner.batch_count, 0);
    float              scale_a{2.f};

    for(int64_t b = 0; b < runner.batch_count; ++b)
    {
        const auto    batch_stride_a = runner.m * runner.k;
        const auto    batch_stride_b = runner.k * runner.n;
        const auto    batch_stride_c = runner.m * runner.n;
        const auto    batch_stride_d = runner.m * runner.n;
        const TypeA*  a_ptr          = reinterpret_cast<const TypeA*>(runner.a);
        const TypeB*  b_ptr          = reinterpret_cast<const TypeB*>(runner.b);
        const TypeCD* c_ptr          = reinterpret_cast<const TypeCD*>(runner.c);
        for(int64_t i = 0; i < runner.m; ++i)
        {
            for(int64_t j = 0; j < runner.n; ++j)
            {
                for(int64_t k = 0; k < runner.k; ++k)
                {
                    ref[batch_stride_d * b + j * runner.m + i]
                        += scale_a * float(a_ptr[batch_stride_a * b + runner.m * k + i])
                           * float(b_ptr[batch_stride_b * b + runner.k * j + k]);
                }

                ref[batch_stride_d * b + j * runner.m + i] *= runner.alpha;
                ref[batch_stride_d * b + j * runner.m + i]
                    += runner.beta * c_ptr[batch_stride_c * b + j * runner.m + i];
            }
        }
    }

    std::vector<TypeCD> gpu_results(runner.m * runner.n * runner.batch_count);
    HIP_CHECK(hipMemcpyDtoH(gpu_results.data(),
                            runner.d_d,
                            runner.batch_count * runner.m * runner.n * sizeof(TypeCD)));

    for(int64_t b = 0; b < runner.batch_count; ++b)
    {
        const auto batch_stride_d = runner.m * runner.n;
        for(int64_t i = 0; i < runner.m; ++i)
        {
            for(int64_t j = 0; j < runner.n; ++j)
            {
                const auto lhs = float(TypeCD(ref[batch_stride_d * b + j * runner.m + i]));
                const auto rhs = float(gpu_results[batch_stride_d * b + j * runner.m + i]);

                if(std::abs(lhs - rhs) > 1e-5)
                {
                    std::cout << lhs << " vs " << rhs << '\n';
                    // assert(ref[batch_stride_d * b + j * runner.m + i] == float(gpu_results[batch_stride_d * b + j * runner.m + i]));
                    return -1;
                }
            }
        }
    }

    return 0;
}

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    runner<hipblaslt_f8_fnuz, hipblasLtHalf, float, float, float>
        runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run(
        [&runner]
        {
            gemm_mix_precision_ext(runner.handle,
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

    if(validate(runner))
    {
        std::cerr << "Validation failed\n";
    }

    return 0;
}

void gemm_mix_precision_ext(hipblasLtHandle_t  handle,
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
    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(handle,
                             trans_a,
                             trans_b,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_16F,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIPBLAS_COMPUTE_32F_FAST_16F);

    // Copy scale_a to device memory
    float scale_a   = 2.f;
    void* d_scale_a = nullptr;
    HIP_CHECK(hipMalloc(&d_scale_a, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_scale_a, &scale_a, sizeof(float), hipMemcpyHostToDevice));

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setScaleA(d_scale_a); // Add scale_a, this is a device pointer.
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    HIPBLASLT_CHECK(gemm.algoGetHeuristic(request_solutions, gemm_pref, heuristic_result));

    if(heuristic_result.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        HIP_CHECK(hipFree(d_scale_a));
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returned_algo_count; i++)
    //     workspace_size = std::max(workspace_size, heuristic_result[i].workspaceSize);
    // HIP_CHECK(hipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize every time when algo changes
    HIPBLASLT_CHECK(gemm.initialize(heuristic_result[0].algo, d_workspace));
    HIPBLASLT_CHECK(gemm.run(stream));

    HIP_CHECK(hipFree(d_scale_a));
    return;
}
