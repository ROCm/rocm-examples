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

#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

using rocwmma::get;
using rocwmma::Log2;
using rocwmma::make_coord2d;

// WMMA block dimensions
constexpr int tile_dim  = 16;
constexpr int t_block_x = 128;

// CPU reference implementation for DLRM forward pass
void dlrm_dot_fwd_cpu(
    float16_t const* input, float16_t* output, uint32_t m, uint32_t k, uint32_t batch_size)
{
    auto batch_offset        = m * k;
    uint output_batch_offset = ((m * (m - 1)) / 2) + k;

    for(uint32_t b = 0; b < batch_size; b++)
    {
        uint output_idx = b * output_batch_offset;

        // Copy MLP to output
        for(uint32_t i = 0; i < k; i++)
        {
            output[output_idx] = input[b * batch_offset + i];
            output_idx++;
        }

        for(uint32_t i = 0; i < m; i++)
        {
            for(uint32_t j = 0; j < m; j++)
            {
                float accum = 0.0f;
                for(uint32_t h = 0; h < k; h++)
                {
                    accum += static_cast<float>(input[b * batch_offset + i * k + h])
                             * static_cast<float>(input[b * batch_offset + j * k + h]);
                }

                if(j < i)
                {
                    output[output_idx] = static_cast<float16_t>(accum);
                    output_idx++;
                }
            }
        }
    }
}

// Simplified DLRM forward device kernel
__global__ void dlrm_dot_fwd(const float16_t* __restrict input,
                             float16_t* __restrict output,
                             float* acc,
                             uint   m,
                             uint   k,
                             uint   b,
                             uint   input_batch_offset,
                             uint   output_batch_offset,
                             uint   acc_batch_offset)
{
    using mapping_a   = rocwmma::MappingUtil<tile_dim, tile_dim, float16_t, row_major>;
    using mapping_b   = rocwmma::MappingUtil<tile_dim, tile_dim, float16_t, col_major>;
    using mapping_acc = rocwmma::MappingUtil<tile_dim, tile_dim, float, row_major>;

    using frag_a = rocwmma::fragment<matrix_a, tile_dim, tile_dim, tile_dim, float16_t, row_major>;
    using frag_b = rocwmma::fragment<matrix_b, tile_dim, tile_dim, tile_dim, float16_t, col_major>;
    using frag_acc = rocwmma::fragment<accumulator, tile_dim, tile_dim, tile_dim, float>;

    (void)b;

    // Copy bottom MLP to output
    auto global_thread_coord = blockIdx.x * blockDim.x + threadIdx.x;
    auto count               = k >> Log2<t_block_x>::value;

    if(blockIdx.x == 0 && blockIdx.y == 0)
    {
        for(uint32_t i = 0; i < count; i++)
        {
            if(i * blockDim.x + global_thread_coord < k)
            {
                output[output_batch_offset * blockIdx.z + i * blockDim.x + global_thread_coord]
                    = input[input_batch_offset * blockIdx.z + i * blockDim.x + global_thread_coord];
            }
        }
    }

    // Target output block
    auto matrix_coord_c = mapping_acc::matrixCoord();

    if(get<0>(matrix_coord_c) < m && get<1>(matrix_coord_c) < m)
    {
        // Initialize accumulator
        auto frag_acc_val = frag_acc();
        rocwmma::fill_fragment(frag_acc_val, static_cast<float>(0));

        // Setup starting addresses
        auto* input_with_offset = input + input_batch_offset * blockIdx.z;
        auto* addr_a
            = mapping_a::dataCoord(input_with_offset, make_coord2d(get<0>(matrix_coord_c), 0), k);
        auto* addr_b
            = mapping_b::dataCoord(input_with_offset, make_coord2d(0, get<1>(matrix_coord_c)), k);

        // Setup address increments
        auto incr_a = mapping_a::dataOffset(make_coord2d(0, tile_dim), k);
        auto incr_b = mapping_b::dataOffset(make_coord2d(tile_dim, 0), k);

        auto count_inner = k / tile_dim;
        for(uint32_t i = 0; i < count_inner; i++)
        {
            rocwmma::synchronize_workgroup();

            auto frag_a_val = frag_a();
            auto frag_b_val = frag_b();

            // Load and multiply
            rocwmma::load_matrix_sync(frag_a_val, addr_a, k);
            rocwmma::load_matrix_sync(frag_b_val, addr_b, k);
            rocwmma::mma_sync(frag_acc_val, frag_a_val, frag_b_val, frag_acc_val);

            addr_a += incr_a;
            addr_b += incr_b;
        }
        rocwmma::synchronize_workgroup();

        // Store frag_acc_val to global acc
        auto* acc_with_offset = acc + acc_batch_offset * blockIdx.z;
        auto* addr_acc        = mapping_acc::dataCoord(acc_with_offset, matrix_coord_c, m);
        rocwmma::store_matrix_sync(addr_acc, frag_acc_val, m, rocwmma::mem_row_major);

        // Copy lower triangular from acc to output
        auto frag_col_idx   = threadIdx.x % tile_dim;
        auto global_col_idx = get<1>(matrix_coord_c) + frag_col_idx;
        auto rows_per_step  = rocwmma::Constants::AMDGCN_WAVE_SIZE / tile_dim;

        auto count_final
            = (tile_dim * tile_dim) >> Log2<rocwmma::Constants::AMDGCN_WAVE_SIZE>::value;
        for(int i = 0; i < count_final; i++)
        {
            auto frag_row_idx
                = i * rows_per_step
                  + ((threadIdx.x & (rocwmma::Constants::AMDGCN_WAVE_SIZE - 1)) / tile_dim);
            auto global_row_idx = get<0>(matrix_coord_c) + frag_row_idx;

            if(global_row_idx > global_col_idx)
            {
                auto output_offset = k + ((global_row_idx * (global_row_idx - 1)) >> 1);
                output[output_batch_offset * blockIdx.z + output_offset + global_col_idx]
                    = float16_t(
                        acc[acc_batch_offset * blockIdx.z + global_row_idx * m + global_col_idx]);
            }
        }
    }
}

int main()
{
    // 1. Set up input data
    constexpr uint32_t m = 32;
    constexpr uint32_t k = 128;
    constexpr uint32_t b = 64;

    // Validate matrix dimensions
    if(m % tile_dim | k % tile_dim)
    {
        std::cerr << "Error: Matrix dimensions not supported" << std::endl;
        return 0;
    }

    std::cout << "rocWMMA Simple DLRM Example" << std::endl;
    std::cout << "Interaction computation: " << m << " embeddings of size " << k
              << " with batch size " << b << std::endl;

    // Calculate sizes
    const size_t tril_size           = ((m * (m - 1)) / 2) + k;
    const uint   input_batch_offset  = m * k;
    const uint   output_batch_offset = tril_size;
    const uint   acc_batch_offset    = m * m;

    // 2. Initialize host matrices
    std::vector<float16_t> h_input(m * k * b);
    std::vector<float16_t> h_output(tril_size * b);

    fill<float16_t>(h_input.data(), m, k, b);

    // 3. Allocate device memory and copy input data
    float16_t* d_input;
    float16_t* d_output;
    float*     d_acc_fwd;

    const size_t input_bytes   = h_input.size() * sizeof(float16_t);
    const size_t output_bytes  = h_output.size() * sizeof(float16_t);
    const size_t acc_fwd_bytes = m * m * b * sizeof(float32_t);

    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_acc_fwd, acc_fwd_bytes));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_bytes, hipMemcpyHostToDevice));

    // 4. Configure kernel launch parameters
    const uint32_t wave_size = get_warp_size();
    auto           grid_dim  = dim3(rocwmma::ceil_div(m, tile_dim * t_block_x / wave_size),
                         rocwmma::ceil_div(m, tile_dim),
                         b);
    auto           block_dim = dim3(t_block_x);

    std::cout << "Launching kernel with grid(" << grid_dim.x << "," << grid_dim.y << ","
              << grid_dim.z << ") block(" << block_dim.x << ")" << std::endl;

    // 5. Launch rocWMMA DLRM kernel
    // Warm-up
    for(uint32_t i = 0; i < 5; ++i)
    {
        dlrm_dot_fwd<<<grid_dim, block_dim>>>(d_input,
                                              d_output,
                                              d_acc_fwd,
                                              m,
                                              k,
                                              b,
                                              input_batch_offset,
                                              output_batch_offset,
                                              acc_batch_offset);
    }

    constexpr uint32_t record_runs = 100u;

    // Actual recorded runs
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    HIP_CHECK(hipEventRecord(start_event));
    for(uint32_t i = 0; i < record_runs; ++i)
    {
        dlrm_dot_fwd<<<grid_dim, block_dim>>>(d_input,
                                              d_output,
                                              d_acc_fwd,
                                              m,
                                              k,
                                              b,
                                              input_batch_offset,
                                              output_batch_offset,
                                              acc_batch_offset);
    }
    HIP_CHECK(hipEventRecord(stop_event));
    HIP_CHECK(hipEventSynchronize(stop_event));

    auto elapsed_time_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_time_ms, start_event, stop_event));

    auto gflops = calculate_gflops(m, m, k) * b;
    auto tflops_per_sec
        = calculate_tflops_per_sec(m, m, k, static_cast<double>(elapsed_time_ms), record_runs * b);

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

    std::cout << std::left << std::setw(8) << t_block_x << std::setw(8) << "N/A" << std::setw(8)
              << "N/A" << std::setw(8) << "N/A" << std::setw(6) << tile_dim << std::setw(6)
              << tile_dim << std::setw(6) << tile_dim << std::setw(8) << m << std::setw(8) << m
              << std::setw(8) << k << std::setw(8) << "N/A" << std::setw(8) << "N/A" << std::setw(8)
              << "N/A" << std::setw(8) << "N/A" << std::setw(8) << "N/A" << std::setw(13)
              << elapsed_time_ms << std::setw(23) << gflops << std::setw(10) << tflops_per_sec
              << std::endl;

    // 6. Copy result back to host
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_bytes, hipMemcpyDeviceToHost));

    // 7. Validate result (CPU reference calculation)
    std::vector<float16_t> output_ref(h_output.size());
    dlrm_dot_fwd_cpu(h_input.data(), output_ref.data(), m, k, b);

    auto res = compare_equal<float16_t>(h_output.data(), output_ref.data(), h_output.size(), 1.0);

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
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_acc_fwd));

    return 0;
}
