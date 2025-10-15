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
constexpr auto computePrecision = HIPSPARSELT_COMPUTE_32F; // set compute precision to 32-bit floating point (HIP only)
#elif defined(__HIP_PLATFORM_NVIDIA__)
constexpr auto computePrecision = HIPSPARSELT_COMPUTE_16F; // set compute precision to 16-bit floating point (CUDA only)
#endif

int main()
{
    std::srand(std::time(nullptr));
    // Generates random values in [0, 1]
    auto randomHalf = []()
    {
        return __float2half(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
    };

    // Create a stream for the matrix multiplication
    auto matmulStream = hipStream_t{};
    HIP_CHECK(hipStreamCreate(&matmulStream));

    // Matrix dimensions and scaling factors
    constexpr auto m = 32;
    constexpr auto n = 32;
    constexpr auto p = 64;

    auto hostAlpha = std::vector<__half>{};
    hostAlpha.resize(p);
    std::generate(std::begin(hostAlpha), std::end(hostAlpha), randomHalf);

    auto deviceAlpha = static_cast<__half*>(nullptr);
    constexpr auto alphaBytes = p * sizeof(__half);
    HIP_CHECK(hipMalloc(&deviceAlpha, alphaBytes));
    HIP_CHECK(hipMemcpy(deviceAlpha, hostAlpha.data(), alphaBytes, hipMemcpyHostToDevice));

    constexpr auto beta = 1.f;

    // Initialize hipSPARSELt
    auto handle = hipsparseLtHandle_t{};
    HIPSPARSELT_CHECK(hipsparseLtInit(&handle));

    // Initialize sparse matrix
    constexpr auto rowsA = m;
    constexpr auto colsA = p;
    constexpr auto ldA = colsA;
    auto ADesc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtStructuredDescriptorInit(
        &handle,
        &ADesc,
        rowsA,                            // number of rows (= column length)
        colsA,                            // number of columns (= row length)
        ldA,                              // leading dimension
        16,                               // alignment (not used by AMD targets)
        HIP_R_16F,                        // datatype (half)
        HIPSPARSE_ORDER_ROW,              // memory layout (row-major)
        HIPSPARSELT_SPARSITY_50_PERCENT   // Sparsity
    ));

    auto hostA = std::vector<__half>{};
    hostA.resize(colsA * rowsA);
    std::generate(std::begin(hostA), std::end(hostA), randomHalf);

    auto deviceA = static_cast<__half*>(nullptr);
    constexpr auto ABytes = colsA * rowsA * sizeof(__half);
    HIP_CHECK(hipMalloc(&deviceA, ABytes));
    HIP_CHECK(hipMemcpy(deviceA, hostA.data(), ABytes, hipMemcpyHostToDevice));

    // Initialize dense matrices -- B will be transposed
    constexpr auto rowsB = n;
    constexpr auto colsB = p;
    constexpr auto ldB = colsB;
    auto BDesc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &BDesc, rowsB, colsB, ldB, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto hostB = std::vector<__half>{};
    hostB.resize(colsB * rowsB);
    std::generate(std::begin(hostB), std::end(hostB), randomHalf);

    auto deviceB = static_cast<__half*>(nullptr);
    constexpr auto BBytes = colsB * rowsB * sizeof(__half);
    HIP_CHECK(hipMalloc(&deviceB, BBytes));
    HIP_CHECK(hipMemcpy(deviceB, hostB.data(), BBytes, hipMemcpyHostToDevice));
   
    constexpr auto rowsC = m;
    constexpr auto colsC = n;
    constexpr auto ldC = colsC;
    auto CDesc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &CDesc, rowsC, colsC, ldC, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto hostC = std::vector<__half>{};
    hostC.resize(colsC * rowsC);
    std::generate(std::begin(hostC), std::end(hostC), randomHalf);

    auto deviceC = static_cast<__half*>(nullptr);
    constexpr auto CBytes = colsC * rowsC * sizeof(__half);
    HIP_CHECK(hipMalloc(&deviceC, CBytes));
    HIP_CHECK(hipMemcpy(deviceC, hostC.data(), CBytes, hipMemcpyHostToDevice));

    constexpr auto rowsD = rowsC;
    constexpr auto colsD = colsC;
    constexpr auto ldD = ldC;
    auto DDesc = hipsparseLtMatDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(
        &handle, &DDesc, rowsD, colsD, ldD, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW
    ));

    auto deviceD = static_cast<__half*>(nullptr);
    constexpr auto DBytes = colsD * rowsD * sizeof(__half);
    HIP_CHECK(hipMalloc(&deviceD, DBytes));
    HIP_CHECK(hipMemset(deviceD, 0, DBytes));

    // Initialize matrix multiplication
    auto matmulDesc = hipsparseLtMatmulDescriptor_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulDescriptorInit(
        &handle,
        &matmulDesc,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,  // do not transpose A
        HIPSPARSE_OPERATION_TRANSPOSE,      // transpose B
        &ADesc,                    
        &BDesc,
        &CDesc,
        &DDesc,
        computePrecision
    ));

    // Set alpha vector mode
    auto alphaMode = 1;
    HIPSPARSELT_CHECK(hipsparseLtMatmulDescSetAttribute(
        &handle,
        &matmulDesc,
        HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
        &alphaMode,
        sizeof(alphaMode)
    ));

    // Select algorithm
    auto matmulAlgSelect = hipsparseLtMatmulAlgSelection_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulAlgSelectionInit(
        &handle, &matmulAlgSelect, &matmulDesc, HIPSPARSELT_MATMUL_ALG_DEFAULT
    ));

    // Initialize plan
    auto matmulPlan = hipsparseLtMatmulPlan_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanInit(
        &handle, &matmulPlan, &matmulDesc, &matmulAlgSelect
    ));

    // Allocate workspace
    auto workspaceSize = std::size_t{};
    HIPSPARSELT_CHECK(hipsparseLtMatmulGetWorkspace(&handle, &matmulPlan, &workspaceSize));
    auto workspace = static_cast<void*>(nullptr);
    if(workspaceSize > 0)
        HIP_CHECK(hipMalloc(&workspace, workspaceSize));

    // Prune A using 2:4 sparsity pattern and verify success
    auto prunedA = static_cast<__half*>(nullptr); // temporary buffer for pruned A
    HIP_CHECK(hipMalloc(&prunedA, ABytes));

    HIPSPARSELT_CHECK(hipsparseLtSpMMAPrune(
        &handle, &matmulDesc, deviceA, prunedA, HIPSPARSELT_PRUNE_SPMMA_TILE, matmulStream
    ));

    auto deviceIsValid = static_cast<int*>(nullptr);
    HIP_CHECK(hipMalloc(&deviceIsValid, sizeof(int)));
    HIPSPARSELT_CHECK(hipsparseLtSpMMAPruneCheck(
        &handle, &matmulDesc, prunedA, deviceIsValid, matmulStream
    ));

    auto hostIsValid = int{};
    HIP_CHECK(hipMemcpyAsync(&hostIsValid, deviceIsValid, sizeof(int), hipMemcpyDeviceToHost, matmulStream));
    HIP_CHECK(hipStreamSynchronize(matmulStream));
    if(hostIsValid != 0) // 0 correct, 1 wrong
    {
        std::cerr << "Error: Matrix pruning failed to achieve required sparsity pattern." << std::endl;
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipFree(deviceIsValid));
    
    // Compress pruned A
    auto compressedA = static_cast<__half*>(nullptr);
    auto compressBuf = static_cast<__half*>(nullptr); // temporary buffer for compression
    auto compressedASize = std::size_t{};
    auto compressBufSize = std::size_t{};
    HIPSPARSELT_CHECK(hipsparseLtSpMMACompressedSize(&handle, &matmulPlan, &compressedASize, &compressBufSize));

    HIP_CHECK(hipMalloc(&compressedA, compressedASize));
    HIP_CHECK(hipMalloc(&compressBuf, compressBufSize));

    HIPSPARSELT_CHECK(hipsparseLtSpMMACompress(
        &handle, &matmulPlan, prunedA, compressedA, compressBuf, matmulStream
    ));

    // Compressed A can now be used - clean up temporary buffers
    HIP_CHECK(hipFree(compressBuf));
    HIP_CHECK(hipFree(prunedA));

    // Perform the matrix multiplication: D = α^T × A × B^T + β × C on a single stream
    HIPSPARSELT_CHECK(hipsparseLtMatmul(
        &handle, &matmulPlan, deviceAlpha, compressedA, deviceB, &beta, deviceC, deviceD, workspace, &matmulStream, 1
    ));

    // Wait for the work to finish
    HIP_CHECK(hipStreamSynchronize(matmulStream));

    // Copy result to host
    auto hostD = std::vector<__half>{};
    hostD.resize(colsD * rowsD);
    HIP_CHECK(hipMemcpy(hostD.data(), deviceD, DBytes, hipMemcpyDeviceToHost));

    // Clean up
    HIP_CHECK(hipFree(compressedA));
    HIP_CHECK(hipFree(workspace));
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanDestroy(&matmulPlan));
    HIP_CHECK(hipFree(deviceD));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&DDesc));
    HIP_CHECK(hipFree(deviceC));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&CDesc));
    HIP_CHECK(hipFree(deviceB));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&BDesc));
    HIP_CHECK(hipFree(deviceA));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&ADesc));
    HIPSPARSELT_CHECK(hipsparseLtDestroy(&handle));
    HIP_CHECK(hipFree(deviceAlpha));
    HIP_CHECK(hipStreamDestroy(matmulStream));

    return EXIT_SUCCESS;
}
