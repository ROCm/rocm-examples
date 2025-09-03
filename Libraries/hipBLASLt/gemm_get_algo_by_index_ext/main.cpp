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
#include <numeric>

void gemm_get_algo_by_index_ext(hipblasLtHandle_t  handle,
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
            gemm_get_algo_by_index_ext(runner.handle,
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

void gemm_get_algo_by_index_ext(hipblasLtHandle_t  handle,
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

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    int                                           algo_index_count = 0;
    int                                           algo_index_inc   = 100;
    while(1)
    {
        // Get algos by index
        // This sample uses a while loop to search for any solution that fits the problem.
        // In real cases, the user can use the saved algo index to get the algorithm.
        // isAlgoSupported is not necessary if the user is sure that the algo supports the problem.

        // The API supports get multiple solutions with a vector.
        std::vector<int> algo_index(algo_index_inc);
        std::iota(std::begin(algo_index), std::end(algo_index), algo_index_count);
        algo_index_count += algo_index_inc;
        std::vector<hipblasLtMatmulHeuristicResult_t> test_results;
        if(HIPBLAS_STATUS_INVALID_VALUE
           == hipblaslt_ext::getAlgosFromIndex(handle, algo_index, test_results))
        {
            std::cout << "Indexes are all out of bound." << std::endl;
            break;
        }
        bool found_algo = false;
        for(size_t i = 0; i < test_results.size(); i++)
        {
            size_t workspace_size_in_bytes = 0;
            size_t workspace_size          = 0;
            if(gemm.isAlgoSupported(test_results[i].algo, workspace_size_in_bytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspace_size_in_bytes <= (size_t)max_workspace_size)
                {
                    workspace_size = std::max(workspace_size, workspace_size_in_bytes);
                    std::cout << "Algo index found: "
                              << hipblaslt_ext::getIndexFromAlgo(test_results[i].algo) << std::endl;
                    heuristic_result.push_back(test_results[i]);
                    found_algo = true;
                    break;
                }
            }
        }
        if(found_algo)
            break;
    }

    if(heuristic_result.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // HIP_CHECK(hipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize every time when algo changes
    HIPBLASLT_CHECK(gemm.initialize(heuristic_result[0].algo, d_workspace));
    HIPBLASLT_CHECK(gemm.run(stream));
    return;
}
