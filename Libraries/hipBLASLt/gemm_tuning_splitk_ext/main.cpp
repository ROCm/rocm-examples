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

void gemm_tuning_splitk_ext(hipblasLtHandle_t  handle,
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
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run(
        [&runner]
        {
            gemm_tuning_splitk_ext(runner.handle,
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

void gemm_tuning_splitk_ext(hipblasLtHandle_t  handle,
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
    // Get all algo doesn't require a gemm instance.
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
                                               hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                               trans_a,
                                               trans_a,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIPBLAS_COMPUTE_32F,
                                               heuristic_result));

    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(handle,
                             trans_a,
                             trans_b,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    std::vector<hipblaslt_ext::GemmTuning> tunings;
    tunings.resize(2);
    tunings[1].setSplitK(8);
    // Not all the solutions supports GemmTuning, if you create a
    // hipblaslt_ext::GemmTuning without changing any default values,
    // the effect is same as calling API
    // isAlgoSupported(algo, returnedWorkspaceSize)

    uint64_t            workspace_size = 0;
    std::vector<size_t> valid_idx;
    std::vector<size_t> valid_idx_tuning;
    for(size_t i = 0; i < heuristic_result.size(); i++)
    {
        size_t workspace_size_in_bytes = 0;
        // If tuning is given, the API will not return success if the solution cannot
        // accept an user tuning parameter.
        if(gemm.isAlgoSupported(heuristic_result[i].algo, tunings[0], workspace_size_in_bytes)
           == HIPBLAS_STATUS_SUCCESS)
        {
            if(workspace_size_in_bytes <= (size_t)max_workspace_size)
            {
                workspace_size = std::max(workspace_size, workspace_size_in_bytes);
                valid_idx.push_back(i);
            }
        }
        if(gemm.isAlgoSupported(heuristic_result[i].algo, tunings[1], workspace_size_in_bytes)
           == HIPBLAS_STATUS_SUCCESS)
        {
            if(workspace_size_in_bytes <= (size_t)max_workspace_size)
            {
                workspace_size = std::max(workspace_size, workspace_size_in_bytes);
                valid_idx_tuning.push_back(i);
            }
        }
    }

    if(valid_idx.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }
    if(valid_idx_tuning.empty())
    {
        std::cerr << "No valid tuning solution found!" << std::endl;
        return;
    }
    // Note that different Tuning configurations will get different
    // amounts of valid_idx.

    void* ws_ptr = nullptr;
    // Changing GSU might require more workspace_size.
    if(workspace_size > (uint64_t)max_workspace_size)
    {
        static_cast<void>(hipMalloc(&ws_ptr, workspace_size));
    }
    else
    {
        ws_ptr = d_workspace;
    }

    HIPBLASLT_CHECK(gemm.initialize(heuristic_result[valid_idx[0]].algo, tunings[0], ws_ptr));
    HIPBLASLT_CHECK(gemm.run(stream));

    // Make sure to initialize every time when algo changes
    // If tuning is given, the API will not return success if the solution cannot accept an user tuning parameter.
    HIPBLASLT_CHECK(
        gemm.initialize(heuristic_result[valid_idx_tuning[0]].algo, tunings[1], ws_ptr));
    HIPBLASLT_CHECK(gemm.run(stream));

    if(workspace_size > (uint64_t)max_workspace_size)
    {
        static_cast<void>(hipFree(ws_ptr));
    }
    return;
}
