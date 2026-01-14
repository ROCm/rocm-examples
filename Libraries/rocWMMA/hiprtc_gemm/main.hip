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

#include "rocwmma_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using rocwmma::bfloat16_t;
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;

using rocwmma::col_major;
using rocwmma::row_major;

// Types
using input_t   = bfloat16_t;
using output_t  = float32_t;
using compute_t = float32_t;

// WMMA block dimensions
const int rocwmma_m = 16;
const int rocwmma_n = 16;
const int rocwmma_k = 16;

// Device warp size
const int wave_size = get_warp_size();

// Thread block
const int t_block_x = 4 * wave_size;
const int t_block_y = 4;

std::string source = R"(

#include <cstdint>
#include <rocwmma/rocwmma.hpp>

using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;
using rocwmma::bfloat16_t;
using rocwmma::uint32_t;

using input_t   = bfloat16_t;
using output_t  = float32_t;
using compute_t = float32_t;

constexpr int rocwmma_m = 16;
constexpr int rocwmma_n = 16;
constexpr int rocwmma_k = 16;

extern "C"
__global__ void gemm_rocwmma_d(uint32_t         m,
                               uint32_t         n,
                               uint32_t         k,
                               input_t const* a,
                               input_t const* b,
                               output_t const* c,
                               output_t*       d,
                               uint32_t         lda,
                               uint32_t         ldb,
                               uint32_t         ldc,
                               uint32_t         ldd,
                               compute_t        alpha,
                               compute_t        beta)
{
    // Create frags
    auto frag_a = rocwmma::fragment<rocwmma::matrix_a,
                                   rocwmma_m,
                                   rocwmma_n,
                                   rocwmma_k,
                                   input_t,
                                   rocwmma::row_major>();
    auto frag_b = rocwmma::fragment<rocwmma::matrix_b,
                                   rocwmma_m,
                                   rocwmma_n,
                                   rocwmma_k,
                                   input_t,
                                   rocwmma::col_major>();
    auto frag_c
        = rocwmma::fragment<rocwmma::accumulator, rocwmma_m, rocwmma_n, rocwmma_k, output_t>();
    auto frag_acc
        = rocwmma::fragment<rocwmma::accumulator, rocwmma_m, rocwmma_n, rocwmma_k, compute_t>();

    rocwmma::fill_fragment(frag_acc, 0.0f);

    // Tile using a 2D grid
    auto major_warp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minor_warp = (blockIdx.y * blockDim.y + threadIdx.y);

    // Target C block
    auto c_row = major_warp * rocwmma_m;
    auto c_col = minor_warp * rocwmma_n;

    // Bounds check
    if(c_row < m && c_col < n)
    {
        // frag_acc = A x B
        for(int i = 0; i < k; i += rocwmma_k)
        {
            // Load the inputs
            rocwmma::load_matrix_sync(frag_a, a + (c_row * lda + i), lda);
            rocwmma::load_matrix_sync(frag_b, b + (i + c_col * ldb), ldb);

            // Matrix multiply - accumulate using MFMA units
            rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }

        // Fetch C matrix
        rocwmma::load_matrix_sync(frag_c, c + (c_row * ldc + c_col), ldc, rocwmma::mem_row_major);

        // D = alpha * A x B + beta * C
        for(int i = 0; i < frag_c.num_elements; ++i)
        {
            frag_c.x[i] = alpha * frag_acc.x[i] + beta * frag_c.x[i];
        }

        // Store to D
        rocwmma::store_matrix_sync(d + (c_row * ldd + c_col), frag_c, ldd, rocwmma::mem_row_major);
    }
}
)";

char const* src = source.c_str();

int main()
{
    // 1. Set up input data
    uint32_t  m     = 256;
    uint32_t  n     = 256;
    uint32_t  k     = 256;
    compute_t alpha = 2.1f;
    compute_t beta  = 2.1f;

    // Determine the rocm path to use for build
    std::string rocm_path
        = (std::getenv("ROCM_PATH") != nullptr) ? std::getenv("ROCM_PATH") : ROCM_ROOT_PATH;
    std::string rocwmma_include_path = std::string("-I") + rocm_path + std::string("/include");

    // 2. Compile kernel with hipRTC
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, src, nullptr, 0, nullptr, nullptr));
    const char* opts[] = {"-D__HIP_PLATFORM_AMD__", "--std=c++17", rocwmma_include_path.c_str()};

    hiprtcResult result = hiprtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts);
    if(result != HIPRTC_SUCCESS)
    {
        std::cout << "HipRTC compile failed." << std::endl;
        std::cout << result << std::endl;
        std::string s_error = hiprtcGetErrorString(result);
        std::cout << s_error << std::endl;
        std::size_t log_size;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &log_size));
        std::cout << "Log Size: " << log_size << std::endl;
        std::string log;

        log.reserve(log_size);
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));

        std::cout << log.c_str() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 3. Load compiled module and get function
    std::size_t code_size;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);

    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

    hipModule_t   module;
    hipFunction_t func;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&func, module, "gemm_rocwmma_d"));

    // Bounds check
    if((m < (uint32_t)(rocwmma_m * t_block_x / wave_size) || n < (rocwmma_n * t_block_y)
        || k < rocwmma_k)
       || (m % rocwmma_m || n % rocwmma_n || k % rocwmma_k))
    {
        std::cout << "Unsupported size!\n";
        return 0;
    }

    uint32_t lda = k;
    uint32_t ldb = k;
    uint32_t ldc = n;
    uint32_t ldd = ldc;

    std::cout << "rocWMMA hipRTC GEMM Example" << std::endl;
    std::cout << "Initializing host data..." << std::endl;

    // 4. Initialize host matrices
    std::vector<input_t>  matrix_a(m * k);
    std::vector<input_t>  matrix_b(k * n);
    std::vector<output_t> matrix_c(m * n);
    std::vector<output_t> matrix_d(m * n, std::numeric_limits<output_t>::signaling_NaN());

    fill_rand(matrix_a.data(), m, k);
    fill_rand(matrix_b.data(), k, n);
    fill_rand(matrix_c.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // 5. Allocate device memory and copy input data
    hipDeviceptr_t d_a, d_b, d_c, d_d;

    const size_t bytes_a = matrix_a.size() * sizeof(input_t);
    const size_t bytes_b = matrix_b.size() * sizeof(input_t);
    const size_t bytes_c = matrix_c.size() * sizeof(output_t);
    const size_t bytes_d = matrix_d.size() * sizeof(output_t);

    HIP_CHECK(hipMalloc(&d_a, bytes_a));
    HIP_CHECK(hipMalloc(&d_b, bytes_b));
    HIP_CHECK(hipMalloc(&d_c, bytes_c));
    HIP_CHECK(hipMalloc(&d_d, bytes_d));

    HIP_CHECK(hipMemcpy(d_a, matrix_a.data(), bytes_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, matrix_b.data(), bytes_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, matrix_c.data(), bytes_c, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, matrix_d.data(), bytes_d, hipMemcpyHostToDevice));

    // 6. Configure kernel launch parameters
    auto block_dim = dim3(t_block_x, t_block_y);
    auto grid_dim  = dim3(rocwmma::ceil_div(m, rocwmma_m * t_block_x / wave_size),
                         rocwmma::ceil_div(n, rocwmma_n * t_block_y));

    struct
    {
        uint32_t       _m;
        uint32_t       _n;
        uint32_t       _k;
        hipDeviceptr_t _d_a;
        hipDeviceptr_t _d_b;
        hipDeviceptr_t _d_c;
        hipDeviceptr_t _d_d;
        uint32_t       _lda;
        uint32_t       _ldb;
        uint32_t       _ldc;
        uint32_t       _ldd;
        compute_t      _alpha;
        compute_t      _beta;
    } args{m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta};

    std::size_t args_size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size,
                      HIP_LAUNCH_PARAM_END};

    std::cout << "Launching GEMM kernel..." << std::endl;

    // 7. Launch rocWMMA kernel
    // Warm-up
    for(uint32_t i = 0; i < 5; ++i)
    {
        HIP_CHECK(hipModuleLaunchKernel(func,
                                        grid_dim.x,
                                        grid_dim.y,
                                        grid_dim.z,
                                        block_dim.x,
                                        block_dim.y,
                                        block_dim.z,
                                        0,
                                        nullptr,
                                        nullptr,
                                        (void**)&config));
    }

    constexpr uint32_t record_runs = 100u;

    // Actual recorded runs
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    HIP_CHECK(hipEventRecord(start_event));
    for(uint32_t i = 0; i < record_runs; ++i)
    {
        HIP_CHECK(hipModuleLaunchKernel(func,
                                        grid_dim.x,
                                        grid_dim.y,
                                        grid_dim.z,
                                        block_dim.x,
                                        block_dim.y,
                                        block_dim.z,
                                        0,
                                        nullptr,
                                        nullptr,
                                        (void**)&config));
    }
    HIP_CHECK(hipEventRecord(stop_event));
    HIP_CHECK(hipEventSynchronize(stop_event));

    auto elapsed_time_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_time_ms, start_event, stop_event));

    auto gflops = calculate_gflops(m, n, k);
    auto tflops_per_sec
        = calculate_tflops_per_sec(m, n, k, static_cast<double>(elapsed_time_ms), record_runs);

    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));

    // Echo performance
    std::cout << std::left << std::setw(8) << "TBlockX" << std::setw(8) << "TBlockY" << std::setw(8)
              << "BlocksM" << std::setw(8) << "BlocksN" << std::setw(6) << "BlkM" << std::setw(6)
              << "BlkN" << std::setw(6) << "BlkK" << std::setw(8) << "MatM" << std::setw(8)
              << "MatN" << std::setw(8) << "MatK" << std::setw(8) << "alpha" << std::setw(8)
              << "lda" << std::setw(8) << "ldb" << std::setw(8) << "beta" << std::setw(8) << "ldc"
              << std::setw(8) << "ldd" << std::setw(13) << "elapsedMs" << std::setw(23)
              << "Problem Size(GFlops)" << std::setw(10) << "TFlops/s" << std::endl;

    std::cout << std::left << std::setw(8) << t_block_x << std::setw(8) << t_block_y << std::setw(8)
              << "N/A" << std::setw(8) << "N/A" << std::setw(6) << rocwmma_m << std::setw(6)
              << rocwmma_n << std::setw(6) << rocwmma_k << std::setw(8) << m << std::setw(8) << n
              << std::setw(8) << k << std::setw(8) << alpha << std::setw(8) << lda << std::setw(8)
              << ldb << std::setw(8) << beta << std::setw(8) << ldc << std::setw(8) << ldd
              << std::setw(13) << elapsed_time_ms << std::setw(23) << gflops << std::setw(10)
              << tflops_per_sec << std::endl;

    // 8. Copy result back to host
    HIP_CHECK(hipMemcpy(matrix_d.data(), d_d, bytes_d, hipMemcpyDeviceToHost));

    // 9. Validate result (CPU reference calculation)
    std::cout << "Validating result with reference..." << std::endl;
    std::vector<output_t> matrix_d_ref(m * n, std::numeric_limits<output_t>::signaling_NaN());
    gemm_cpu_h<input_t, output_t, compute_t, row_major, col_major, row_major>(m,
                                                                              n,
                                                                              k,
                                                                              matrix_a.data(),
                                                                              matrix_b.data(),
                                                                              matrix_c.data(),
                                                                              matrix_d_ref.data(),
                                                                              lda,
                                                                              ldb,
                                                                              ldc,
                                                                              ldd,
                                                                              alpha,
                                                                              beta);

    auto res = compare_equal<output_t>(matrix_d.data(), matrix_d_ref.data(), m * n);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED!" << std::endl;
        std::cout << "Max relative error: " << std::get<1>(res) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "PASSED!" << std::endl;
    }

    // 10. Clean up device memory and hipRTC resources
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_d));

    std::cout << "Finished!" << std::endl;

    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    return 0;
}
