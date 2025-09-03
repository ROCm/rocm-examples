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

void gemm_gelu_aux_bias_ext(hipblasLtHandle_t   handle,
                            hipblasOperation_t  trans_a,
                            hipblasOperation_t  trans_b,
                            hipblasLtEpilogue_t epilogue,
                            int64_t             m,
                            int64_t             n,
                            int64_t             k,
                            int64_t             batch_count,
                            float&              alpha,
                            float&              beta,
                            void*               d_a,
                            void*               d_b,
                            void*               d_c,
                            void*               d_d,
                            void*               d_bias_vec,
                            void*               d_workspace,
                            int64_t             max_workspace_size,
                            hipStream_t         stream);

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with epilogue HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
     *  A,B,C,D and Bias are all bf16
     */
    runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
        runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.set_bias_info(true, 'A');

    runner.run(
        [&runner]
        {
            gemm_gelu_aux_bias_ext(runner.handle,
                                   HIPBLAS_OP_N,
                                   HIPBLAS_OP_N,
                                   HIPBLASLT_EPILOGUE_GELU_AUX_BIAS,
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
                                   runner.d_bias_vec,
                                   runner.d_workspace,
                                   runner.max_workspace_size,
                                   runner.stream);
        });

    return 0;
}

void gemm_gelu_aux_bias_ext(hipblasLtHandle_t   handle,
                            hipblasOperation_t  trans_a,
                            hipblasOperation_t  trans_b,
                            hipblasLtEpilogue_t epilogue,
                            int64_t             m,
                            int64_t             n,
                            int64_t             k,
                            int64_t             batch_count,
                            float&              alpha,
                            float&              beta,
                            void*               d_a,
                            void*               d_b,
                            void*               d_c,
                            void*               d_d,
                            void*               d_bias_vec,
                            void*               d_workspace,
                            int64_t             max_workspace_size,
                            hipStream_t         stream)
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

    hipblaslt_ext::GemmEpilogue gemm_epilogue;
    gemm_epilogue.setMode(epilogue);
    gemm_epilogue.setBiasDataType(HIP_R_16F);
    gemm_epilogue.setAuxLeadingDimension(m);
    gemm_epilogue.setAuxBatchStride(m * n);

    // Set auxiliary buffer
    void* d_aux_buffer;
    HIP_CHECK(hipMalloc(&d_aux_buffer, m * n * sizeof(hipblasLtHalf)));

    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setBias(d_bias_vec);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setAux(d_aux_buffer);
    gemm.setProblem(m, n, k, batch_count, gemm_epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    HIPBLASLT_CHECK(gemm.algoGetHeuristic(request_solutions, gemm_pref, heuristic_result));

    if(heuristic_result.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
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
    HIP_CHECK(hipFree(d_aux_buffer));
    return;
}
