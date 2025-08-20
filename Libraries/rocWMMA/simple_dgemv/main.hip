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

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::float64_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

// WMMA block dimensions
constexpr int rocwmma_m = 16;
constexpr int rocwmma_n = 16;
constexpr int rocwmma_k = 16;

// CPU reference implementation for DGEMV
__host__ void dgemv_cpu_h(uint32_t         m,
                          uint32_t         n,
                          uint32_t         k,
                          float64_t const* a,
                          float64_t const* b,
                          float64_t*       c,
                          float64_t        alpha,
                          float64_t        beta)
{
    uint32_t lda = m;
    (void)n;

    for(uint32_t i = 0; i < m; ++i)
    {
        float64_t accum = 0.0;
        for(uint32_t h = 0; h < k; ++h)
        {
            accum += a[h * lda + i] * b[h];
        }
        c[i] = alpha * accum + beta * c[i];
    }
}

// Device kernel implementation
__global__ void dgemv_rocwmma_d(uint32_t         m,
                                uint32_t         n,
                                uint32_t         k,
                                float64_t const* a,
                                float64_t const* b,
                                float64_t*       c,
                                float64_t*       d,
                                uint32_t         lda,
                                uint32_t         ldb,
                                uint32_t         ldc,
                                uint32_t         ldd,
                                float64_t        alpha,
                                float64_t        beta)
{
    // Create fragments
    auto frag_a
        = rocwmma::fragment<matrix_a, rocwmma_m, rocwmma_n, rocwmma_k, float64_t, col_major>();
    auto frag_b
        = rocwmma::fragment<matrix_b, rocwmma_m, rocwmma_n, rocwmma_k, float64_t, col_major>();
    auto frag_c   = rocwmma::fragment<accumulator, rocwmma_m, rocwmma_n, rocwmma_k, float64_t>();
    auto frag_acc = rocwmma::fragment<accumulator, rocwmma_m, rocwmma_n, rocwmma_k, float64_t>();

    (void)n;
    (void)ldd;

    rocwmma::fill_fragment(frag_acc, 0.0);

    int major_warp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    uint32_t c_row = major_warp * rocwmma_m;

    if(c_row < m)
    {
        // Accumulate A x B over K dimension
        for(uint32_t i = 0; i < k; i += rocwmma_k)
        {
            rocwmma::load_matrix_sync(frag_a, a + (c_row + i * lda), lda);
            rocwmma::load_matrix_sync(frag_b, b + i, ldb);
            rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }

        // Load C vector
        rocwmma::load_matrix_sync(frag_c, c + c_row, ldc, rocwmma::mem_col_major);

        // D = alpha * A x B + beta * C
        for(uint32_t i = 0; i < frag_c.num_elements; i++)
        {
            frag_c.x[i] = alpha * frag_acc.x[i] + beta * frag_c.x[i];
        }

        // Store result
        rocwmma::store_matrix_sync(d + c_row, frag_c, ldc, rocwmma::mem_col_major);
    }
}

int main()
{
    // 1. Set up input data
    constexpr uint32_t m = 256;
    constexpr uint32_t k = 256;

    // Thread block configuration
    const uint32_t wave_size = get_warp_size();
    const int      t_block_x = 16 * wave_size;
    const int      t_block_y = 1;
    const uint32_t n         = t_block_y * rocwmma_n;

    constexpr float64_t alpha = 2.1;
    constexpr float64_t beta  = 2.1;

    // Check device support for F64 operations
    if(!is_f64_supported())
    {
        std::cout << "F64 DGEMV not supported on this device" << std::endl;
        return 0;
    }

    // Matrix layouts and leading dimensions
    // A: col-major (M x K), B: col-major (K x 1), C,D: col-major (M x 1)
    int lda = m; // col-major A
    int ldb = k; // col-major B (vector)
    int ldc = m; // col-major C (vector)
    int ldd = ldc;

    // Validate matrix dimensions
    if(m % rocwmma_m | n % rocwmma_n | k % rocwmma_k)
    {
        std::cerr << "Error: Matrix dimensions not supported" << std::endl;
        return 0;
    }

    std::cout << "rocWMMA Simple DGEMV Example" << std::endl;
    std::cout << "Matrix-vector: A(" << m << "x" << k << ") * B(" << k << "x1) + C(" << m
              << "x1) = D(" << m << "x1)" << std::endl;

    // 2. Initialize host matrices
    std::vector<float64_t> matrix_a(m * k); // matrix
    std::vector<float64_t> matrix_b(k * 1); // vector
    std::vector<float64_t> matrix_c(m * 1, 1.0); // accumulator
    std::vector<float64_t> matrix_d(m * 1); // result

    fill_rand(matrix_a.data(), m, k);
    fill_rand(matrix_b.data(), k, 1);

    // 3. Allocate device memory and copy input data
    float64_t* d_a;
    float64_t* d_b;
    float64_t* d_c;
    float64_t* d_d;

    const size_t bytes_a = matrix_a.size() * sizeof(float64_t);
    const size_t bytes_b = matrix_b.size() * sizeof(float64_t);
    const size_t bytes_c = matrix_c.size() * sizeof(float64_t);

    HIP_CHECK(hipMalloc(&d_a, bytes_a));
    HIP_CHECK(hipMalloc(&d_b, bytes_b));
    HIP_CHECK(hipMalloc(&d_c, bytes_c));
    HIP_CHECK(hipMalloc(&d_d, bytes_c));

    HIP_CHECK(hipMemcpy(d_a, matrix_a.data(), bytes_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, matrix_b.data(), bytes_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, matrix_c.data(), bytes_c, hipMemcpyHostToDevice));

    // 4. Configure kernel launch parameters
    auto block_dim = dim3(t_block_x, t_block_y);
    auto grid_dim  = dim3(rocwmma::ceil_div(m, rocwmma_m * t_block_x / wave_size),
                         rocwmma::ceil_div(n, rocwmma_n * t_block_y));

    std::cout << "Launching kernel with grid(" << grid_dim.x << "," << grid_dim.y << ") block("
              << block_dim.x << "," << block_dim.y << ")" << std::endl;

    // 5. Launch rocWMMA DGEMV kernel
    // Warm-up
    for(uint32_t i = 0; i < 5; ++i)
    {
        dgemv_rocwmma_d<<<grid_dim, block_dim>>>(m,
                                                 n,
                                                 k,
                                                 d_a,
                                                 d_b,
                                                 d_c,
                                                 d_d,
                                                 lda,
                                                 ldb,
                                                 ldc,
                                                 ldd,
                                                 alpha,
                                                 beta);
    }

    constexpr uint32_t record_runs = 100u;

    // Actual recorded runs
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    HIP_CHECK(hipEventRecord(start_event));
    for(uint32_t i = 0; i < record_runs; ++i)
    {
        dgemv_rocwmma_d<<<grid_dim, block_dim>>>(m,
                                                 n,
                                                 k,
                                                 d_a,
                                                 d_b,
                                                 d_c,
                                                 d_d,
                                                 lda,
                                                 ldb,
                                                 ldc,
                                                 ldd,
                                                 alpha,
                                                 beta);
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

    // 6. Copy result back to host
    HIP_CHECK(hipMemcpy(matrix_d.data(), d_d, bytes_c, hipMemcpyDeviceToHost));

    // 7. Validate result (CPU reference calculation)
    std::vector<float64_t> matrix_d_ref(matrix_c); // copy original C
    dgemv_cpu_h(m, n, k, matrix_a.data(), matrix_b.data(), matrix_d_ref.data(), alpha, beta);

    auto res = compare_equal<float64_t>(matrix_d.data(), matrix_d_ref.data(), m);

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

    // 8. Clean up device memory
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_d));

    return 0;
}
