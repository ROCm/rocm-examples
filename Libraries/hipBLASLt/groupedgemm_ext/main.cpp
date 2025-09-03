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

void grouped_gemm_ext(hipblasLtHandle_t     handle,
                      hipblasOperation_t    trans_a,
                      hipblasOperation_t    trans_b,
                      std::vector<int64_t>& m,
                      std::vector<int64_t>& n,
                      std::vector<int64_t>& k,
                      std::vector<int64_t>& batch_count,
                      std::vector<float>&   alpha,
                      std::vector<float>&   beta,
                      std::vector<void*>&   d_a,
                      std::vector<void*>&   d_b,
                      std::vector<void*>&   d_c,
                      std::vector<void*>&   d_d,
                      void*                 d_workspace,
                      int64_t               max_workspace_size,
                      hipStream_t           stream);

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    std::vector<int64_t> m           = {1024, 512};
    std::vector<int64_t> n           = {512, 512};
    std::vector<int64_t> k           = {1920, 128};
    std::vector<int64_t> batch_count = {1, 1};
    std::vector<float>   alpha       = {1.0f, 1.0f};
    std::vector<float>   beta        = {1.0f, 1.0f};
    runner_vec<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        runner(m, n, k, batch_count, alpha, beta, 32 * 1024 * 1024);

    runner.run(
        [&runner]
        {
            grouped_gemm_ext(runner.handle,
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

void grouped_gemm_ext(hipblasLtHandle_t     handle,
                      hipblasOperation_t    trans_a,
                      hipblasOperation_t    trans_b,
                      std::vector<int64_t>& m,
                      std::vector<int64_t>& n,
                      std::vector<int64_t>& k,
                      std::vector<int64_t>& batch_count,
                      std::vector<float>&   alpha,
                      std::vector<float>&   beta,
                      std::vector<void*>&   d_a,
                      std::vector<void*>&   d_b,
                      std::vector<void*>&   d_c,
                      std::vector<void*>&   d_d,
                      void*                 d_workspace,
                      int64_t               max_workspace_size,
                      hipStream_t           stream)
{
    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::GroupedGemm groupedgemm(handle,
                                           trans_a,
                                           trans_b,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLAS_COMPUTE_32F);

    std::vector<hipblaslt_ext::GemmEpilogue> epilogue{
        hipblaslt_ext::
            GemmEpilogue()}; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    std::vector<hipblaslt_ext::GemmInputs> inputs(m.size());
    for(size_t i = 0; i < m.size(); i++)
    {
        inputs[i].setA(d_a[i]);
        inputs[i].setB(d_b[i]);
        inputs[i].setC(d_c[i]);
        inputs[i].setD(d_d[i]);
        inputs[i].setAlpha(&alpha[i]);
        inputs[i].setBeta(&beta[i]);
    }
    // hipblaslt_ext::GemmEpilogue supports broadcasting
    groupedgemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    HIPBLASLT_CHECK(groupedgemm.algoGetHeuristic(request_solutions, gemm_pref, heuristic_result));

    if(heuristic_result.empty())
    {
        std::cerr << "No valid solution found! Please try get_all_algos instead" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returnedAlgoCount; i++)
    //     workspace_size = std::max(workspace_size, heuristic_result[i].workspaceSize);
    // HIP_CHECKhipMalloc(&d_workspace, workspace_size));

    // Get the default values from the grouepdgemm object
    hipblaslt_ext::UserArguments* user_args;
    HIP_CHECK(hipHostMalloc(&user_args, m.size() * sizeof(hipblaslt_ext::UserArguments)));
    groupedgemm.getDefaultValueForDeviceUserArguments(user_args);
    // Copy them to device memory
    hipblaslt_ext::UserArguments* d_user_args;
    HIP_CHECK(hipMalloc(&d_user_args, m.size() * sizeof(hipblaslt_ext::UserArguments)));
    HIP_CHECK(hipMemcpy(d_user_args,
                        user_args,
                        m.size() * sizeof(hipblaslt_ext::UserArguments),
                        hipMemcpyHostToDevice));

    // Make sure to initialize every time when algo changes
    HIPBLASLT_CHECK(groupedgemm.initialize(heuristic_result[0].algo, d_workspace));
    HIPBLASLT_CHECK(groupedgemm.run(d_user_args, stream));

    HIP_CHECK(hipFree(user_args));
    HIP_CHECK(hipFree(d_user_args));
    return;
}
