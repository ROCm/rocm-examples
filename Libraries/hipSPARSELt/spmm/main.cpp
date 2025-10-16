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

#include <example_utils.hpp>
#include <hipsparselt_utils.hpp>

#include <hipsparselt/hipsparselt.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <ctime>

#ifdef __HIP_PLATFORM_AMD__
// set compute precision to 32-bit floating point (HIP only)
constexpr auto compute_precision = HIPSPARSELT_COMPUTE_32F;
#elif defined(__HIP_PLATFORM_NVIDIA__)
// set compute precision to 16-bit floating point (CUDA only)
constexpr auto compute_precision = HIPSPARSELT_COMPUTE_16F;
#endif

int main()
{
    std::srand(std::time(nullptr));
    // Generates random values in [0, 1]
    auto random_half = []()
    {
        return __float2half(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
    };

    // Create a stream for the matrix multiplication
    auto matmul_stream = hipStream_t{};
    HIP_CHECK(hipStreamCreate(&matmul_stream));

    // Matrix dimensions and scaling factors
    constexpr auto m = 32;
    constexpr auto n = 32;
    constexpr auto p = 64;
    constexpr auto alpha = 1.f;
    constexpr auto beta = 1.f;

    // Initialize hipSPARSELt
    auto handle = hipsparseLtHandle_t{};
    HIPSPARSELT_CHECK(hipsparseLtInit(&handle));

    // Initialize sparse matrix
    constexpr auto rows_A = m;
    constexpr auto cols_A = p;
    constexpr auto ld_A = cols_A;
    auto A_desc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtStructuredDescriptorInit(
        &handle,
        &A_desc,
        rows_A,                            // number of rows (= column length)
        cols_A,                            // number of columns (= row length)
        ld_A,                              // leading dimension
        16,                               // alignment (not used by AMD targets)
        HIP_R_16F,                        // datatype (half)
        HIPSPARSE_ORDER_ROW,              // memory layout (row-major)
        HIPSPARSELT_SPARSITY_50_PERCENT   // Sparsity
    ));

    auto host_A = std::vector<__half>{};
    host_A.resize(cols_A * rows_A);
    std::generate(std::begin(host_A), std::end(host_A), random_half);

    auto device_A = static_cast<__half*>(nullptr);
    constexpr auto A_bytes = cols_A * rows_A * sizeof(__half);
    HIP_CHECK(hipMalloc(&device_A, A_bytes));
    HIP_CHECK(hipMemcpy(device_A, host_A.data(), A_bytes, hipMemcpyHostToDevice));

    // Initialize dense matrices -- B will be transposed
    constexpr auto rows_B = n;
    constexpr auto cols_B = p;
    constexpr auto ld_B = cols_B;
    auto B_desc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &B_desc, rows_B, cols_B, ld_B, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto host_B = std::vector<__half>{};
    host_B.resize(cols_B * rows_B);
    std::generate(std::begin(host_B), std::end(host_B), random_half);

    auto device_B = static_cast<__half*>(nullptr);
    constexpr auto B_bytes = cols_B * rows_B * sizeof(__half);
    HIP_CHECK(hipMalloc(&device_B, B_bytes));
    HIP_CHECK(hipMemcpy(device_B, host_B.data(), B_bytes, hipMemcpyHostToDevice));
   
    constexpr auto rows_C = m;
    constexpr auto cols_C = n;
    constexpr auto ld_C = cols_C;
    auto C_desc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &C_desc, rows_C, cols_C, ld_C, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto host_C = std::vector<__half>{};
    host_C.resize(cols_C * rows_C);
    std::generate(std::begin(host_C), std::end(host_C), random_half);

    auto device_C = static_cast<__half*>(nullptr);
    constexpr auto C_bytes = cols_C * rows_C * sizeof(__half);
    HIP_CHECK(hipMalloc(&device_C, C_bytes));
    HIP_CHECK(hipMemcpy(device_C, host_C.data(), C_bytes, hipMemcpyHostToDevice));

    constexpr auto rows_D = rows_C;
    constexpr auto cols_D = cols_C;
    constexpr auto ld_D = ld_C;
    auto D_desc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &D_desc, rows_D, cols_D, ld_D, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto device_D = static_cast<__half*>(nullptr);
    constexpr auto D_bytes = cols_D * rows_D * sizeof(__half);
    HIP_CHECK(hipMalloc(&device_D, D_bytes));
    HIP_CHECK(hipMemset(device_D, 0, D_bytes));

    // Initialize matrix multiplication
    auto matmul_desc = hipsparseLtMatmulDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulDescriptorInit(
        &handle,
        &matmul_desc,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,  // do not transpose A
        HIPSPARSE_OPERATION_TRANSPOSE,      // transpose B
        &A_desc,                    
        &B_desc,
        &C_desc,
        &D_desc,
        compute_precision
    ));

    // Select algorithm
    auto matmul_alg_select = hipsparseLtMatmulAlgSelection_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulAlgSelectionInit(
        &handle, &matmul_alg_select, &matmul_desc, HIPSPARSELT_MATMUL_ALG_DEFAULT
    ));

    // Initialize plan
    auto matmul_plan = hipsparseLtMatmulPlan_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanInit(
        &handle, &matmul_plan, &matmul_desc, &matmul_alg_select
    ));

    // Allocate workspace
    auto workspace_size = std::size_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulGetWorkspace(&handle, &matmul_plan, &workspace_size));
    auto workspace = static_cast<void*>(nullptr);
    if(workspace_size > 0)
        HIP_CHECK(hipMalloc(&workspace, workspace_size));

    // Prune A using 2:4 sparsity pattern and verify success
    auto pruned_A = static_cast<__half*>(nullptr); // temporary buffer for pruned A
    HIP_CHECK(hipMalloc(&pruned_A, A_bytes));

    HIPSPARSELT_CHECK(hipsparseLtSpMMAPrune(
        &handle, &matmul_desc, device_A, pruned_A, HIPSPARSELT_PRUNE_SPMMA_TILE, matmul_stream
    ));

    auto device_is_valid = static_cast<int*>(nullptr);
    HIP_CHECK(hipMalloc(&device_is_valid, sizeof(int)));
    HIPSPARSELT_CHECK(hipsparseLtSpMMAPruneCheck(
        &handle, &matmul_desc, pruned_A, device_is_valid, matmul_stream
    ));

    auto host_is_valid = int{};
    HIP_CHECK(hipMemcpyAsync(&host_is_valid, device_is_valid, sizeof(int), hipMemcpyDeviceToHost, matmul_stream));
    HIP_CHECK(hipStreamSynchronize(matmul_stream));
    if(host_is_valid != 0) // 0 correct, 1 wrong
    {
        std::cerr << "Error: Matrix pruning failed to achieve required sparsity pattern." << std::endl;
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipFree(device_is_valid));
    
    // Compress pruned A
    auto compressed_A = static_cast<__half*>(nullptr);
    auto compress_buf = static_cast<__half*>(nullptr); // temporary buffer for compression
    auto compressed_A_size = std::size_t{};
    auto compress_buf_size = std::size_t{};
    HIPSPARSELT_CHECK(hipsparseLtSpMMACompressedSize(&handle, &matmul_plan, &compressed_A_size, &compress_buf_size));

    HIP_CHECK(hipMalloc(&compressed_A, compressed_A_size));
    HIP_CHECK(hipMalloc(&compress_buf, compress_buf_size));

    HIPSPARSELT_CHECK(hipsparseLtSpMMACompress(
        &handle, &matmul_plan, pruned_A, compressed_A, compress_buf, matmul_stream
    ));

    // Compressed A can now be used - clean up temporary buffers
    HIP_CHECK(hipFree(compress_buf));
    HIP_CHECK(hipFree(pruned_A));

    // Perform the matrix multiplication: D = α × A × B^T + β × C on a single stream
    HIPSPARSELT_CHECK(hipsparseLtMatmul(
        &handle, &matmul_plan, &alpha, compressed_A, device_B, &beta, device_C, device_D, workspace, &matmul_stream, 1
    ));

    // Wait for the work to finish
    HIP_CHECK(hipStreamSynchronize(matmul_stream));

    // Copy result to host
    auto host_D = std::vector<__half>{};
    host_D.resize(cols_D * rows_D);
    HIP_CHECK(hipMemcpy(host_D.data(), device_D, D_bytes, hipMemcpyDeviceToHost));

    // Clean up
    HIP_CHECK(hipFree(compressed_A));
    HIP_CHECK(hipFree(workspace));
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanDestroy(&matmul_plan));
    HIP_CHECK(hipFree(device_D));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&D_desc));
    HIP_CHECK(hipFree(device_C));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&C_desc));
    HIP_CHECK(hipFree(device_B));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&B_desc));
    HIP_CHECK(hipFree(device_A));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&A_desc));
    HIPSPARSELT_CHECK(hipsparseLtDestroy(&handle));
    HIP_CHECK(hipStreamDestroy(matmul_stream));

    return EXIT_SUCCESS;
}
