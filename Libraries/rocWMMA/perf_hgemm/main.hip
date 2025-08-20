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

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma_transforms.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace rocwmma;

// Types and Data Layouts
using input_t   = float16_t;
using output_t  = float16_t;
using compute_t = float32_t;

using data_layout_a   = col_major;
using data_layout_b   = row_major;
using data_layout_c   = row_major;
using data_layout_lds = col_major;

// Architecture-specific parameters for optimal performance
namespace gfx9_params
{
enum kernel_params : uint32_t
{
    rocwmma_m = 32u,
    rocwmma_n = 32u,
    rocwmma_k = 16u,
    blocks_m  = 2u,
    blocks_n  = 2u,
    tblock_x  = 128u,
    tblock_y  = 2u,
    warp_size = Constants::AMDGCN_WAVE_SIZE_64
};
} // namespace gfx9_params

namespace gfx11_params
{
enum kernel_params : uint32_t
{
    rocwmma_m = 16u,
    rocwmma_n = 16u,
    rocwmma_k = 16u,
    blocks_m  = 4u,
    blocks_n  = 2u,
    tblock_x  = 64u,
    tblock_y  = 2u,
    warp_size = Constants::AMDGCN_WAVE_SIZE_32
};
} // namespace gfx11_params

// Select parameters based on architecture
#if (ROCWMMA_ARCH_GFX9)
using namespace gfx9_params;
#else
using namespace gfx11_params;
#endif

// Warp tile: computed by each warp
constexpr uint32_t warp_tile_m = blocks_m * rocwmma_m;
constexpr uint32_t warp_tile_n = blocks_n * rocwmma_n;
constexpr uint32_t warp_tile_k = rocwmma_k;

// Macro Tile: computed by each thread block (workgroup)
constexpr uint32_t warps_m      = tblock_x / warp_size;
constexpr uint32_t warps_n      = tblock_y;
constexpr uint32_t macro_tile_m = warps_m * warp_tile_m;
constexpr uint32_t macro_tile_n = warps_n * warp_tile_n;
constexpr uint32_t macro_tile_k = rocwmma_k;

// Fragment type definitions
using mma_frag_a
    = fragment<matrix_a, warp_tile_m, warp_tile_n, warp_tile_k, input_t, data_layout_a>;
using mma_frag_b
    = fragment<matrix_b, warp_tile_m, warp_tile_n, warp_tile_k, input_t, data_layout_b>;
using mma_frag_c
    = fragment<accumulator, warp_tile_m, warp_tile_n, warp_tile_k, output_t, data_layout_c>;
using mma_frag_d   = mma_frag_c;
using mma_frag_acc = fragment<accumulator, warp_tile_m, warp_tile_n, warp_tile_k, compute_t>;

// Global read fragments (cooperative)
using coop_scheduler = fragment_scheduler::coop_row_major_2d<tblock_x, tblock_y>;
using gr_frag_a      = fragment<matrix_a,
                                macro_tile_m,
                                macro_tile_n,
                                macro_tile_k,
                                input_t,
                                data_layout_a,
                                coop_scheduler>;
using gr_frag_b      = fragment<matrix_b,
                                macro_tile_m,
                                macro_tile_n,
                                macro_tile_k,
                                input_t,
                                data_layout_b,
                                coop_scheduler>;

// Local write fragments (LDS layout)
using lw_frag_a = apply_data_layout_t<gr_frag_a, data_layout_lds>;
using lw_frag_b = apply_data_layout_t<apply_transpose_t<gr_frag_b>, data_layout_lds>;

// Local read fragments (for MMA)
using lr_frag_a = apply_data_layout_t<mma_frag_a, data_layout_lds>;
using lr_frag_b = apply_data_layout_t<apply_transpose_t<mma_frag_b>, data_layout_lds>;

// Transform helpers
constexpr auto transform_gr_frag_a_to_lw_frag_a = [](gr_frag_a const& gr_frag_a_val)
{ return apply_data_layout<data_layout_lds>(gr_frag_a_val); };

constexpr auto transform_gr_frag_b_to_lw_frag_b = [](gr_frag_b const& gr_frag_b_val)
{ return apply_data_layout<data_layout_lds>(apply_transpose(gr_frag_b_val)); };

constexpr auto transform_lr_frag_a_to_mma_frag_a = [](lr_frag_a const& lr_frag_a_val)
{ return apply_data_layout<data_layout_a>(lr_frag_a_val); };

constexpr auto transform_lr_frag_b_to_mma_frag_b = [](lr_frag_b const& lr_frag_b_val)
{ return apply_data_layout<data_layout_b>(apply_transpose(lr_frag_b_val)); };

// High-performance GEMM kernel with data reuse and latency hiding
ROCWMMA_KERNEL void __launch_bounds__(256) gemm_rocwmma_d(uint32_t        m,
                                                          uint32_t        n,
                                                          uint32_t        k,
                                                          input_t const*  a,
                                                          input_t const*  b,
                                                          output_t const* c,
                                                          output_t*       d,
                                                          uint32_t        lda,
                                                          uint32_t        ldb,
                                                          uint32_t        ldc,
                                                          uint32_t        ldd,
                                                          compute_t       alpha,
                                                          compute_t       beta)
{
    // Tile sizes
    constexpr auto warp_tile_size  = make_coord2d(warp_tile_m, warp_tile_n);
    constexpr auto macro_tile_size = make_coord2d(macro_tile_m, macro_tile_n);

    // Local warp coordinate
    auto local_warp_coord  = make_coord2d(threadIdx.x / warp_size, threadIdx.y);
    auto local_warp_offset = local_warp_coord * warp_tile_size;

    // Global matrix coordinates
    auto macro_tile_coord = make_coord2d(blockIdx.x, blockIdx.y) * macro_tile_size;
    auto warp_tile_coord  = macro_tile_coord + local_warp_offset;

    // Bounds check
    auto warp_tile_bound = warp_tile_coord + warp_tile_size;
    if(get<0>(warp_tile_bound) > m || get<1>(warp_tile_bound) > n)
    {
        return;
    }

    // Global read coordinate transforms
    using gr_frag_a_map_1d = GetDataLayout_t<gr_frag_a>;
    using gr_frag_b_map_1d = GetDataLayout_t<gr_frag_b>;

    auto global_read_offset_a
        = gr_frag_a_map_1d::fromMatrixCoord(make_coord2d(get<0>(macro_tile_coord), 0u), lda);
    auto global_read_offset_b
        = gr_frag_b_map_1d::fromMatrixCoord(make_coord2d(0u, get<1>(macro_tile_coord)), ldb);

    auto k_step_offset_a = gr_frag_a_map_1d::fromMatrixCoord(make_coord2d(0u, macro_tile_k), lda);
    auto k_step_offset_b = gr_frag_b_map_1d::fromMatrixCoord(make_coord2d(macro_tile_k, 0u), ldb);

    // Initial global pre-fetch for double buffering
    gr_frag_a gr_frag_a_val;
    gr_frag_b gr_frag_b_val;

    load_matrix_sync(gr_frag_a_val, a + global_read_offset_a, lda);
    load_matrix_sync(gr_frag_b_val, b + global_read_offset_b, ldb);

    global_read_offset_a += k_step_offset_a;
    global_read_offset_b += k_step_offset_b;

    // Setup LDS addressing for double buffering
    HIP_DYNAMIC_SHARED(void*, local_mem_ptr);
    using lw_frag_a_shape  = GetIOShape_t<lw_frag_a>;
    using lw_frag_b_shape  = GetIOShape_t<lw_frag_b>;
    using lw_frag_a_map_1d = GetDataLayout_t<lw_frag_a>;
    using lw_frag_b_map_1d = GetDataLayout_t<lw_frag_b>;

    constexpr uint32_t lds_width  = macro_tile_k;
    constexpr uint32_t lds_height = lw_frag_a_shape::BlockHeight + lw_frag_b_shape::BlockHeight;
    constexpr uint32_t size_lds   = lds_height * lds_width;
    constexpr uint32_t lds_ld = std::is_same_v<data_layout_lds, row_major> ? lds_width : lds_height;

    auto* lds_ptr_lo = reinterpret_cast<input_t*>(local_mem_ptr);
    auto* lds_ptr_hi = lds_ptr_lo + size_lds;

    // Local write offsets
    auto lds_write_offset_a = 0u;
    auto lds_write_offset_b
        = lw_frag_a_map_1d::fromMatrixCoord(make_coord2d(lw_frag_a_shape::BlockHeight, 0u), lds_ld);

    // Local read offsets for MMA fragments
    auto lds_read_offset_a
        = lds_write_offset_a
          + lw_frag_a_map_1d::fromMatrixCoord(make_coord2d(get<0>(local_warp_offset), 0u), lds_ld);
    auto lds_read_offset_b
        = lds_write_offset_b
          + lw_frag_b_map_1d::fromMatrixCoord(make_coord2d(get<1>(local_warp_offset), 0u), lds_ld);

    // Write initial prefetch to LDS
    store_matrix_sync(lds_ptr_lo + lds_write_offset_a,
                      transform_gr_frag_a_to_lw_frag_a(gr_frag_a_val),
                      lds_ld);
    store_matrix_sync(lds_ptr_lo + lds_write_offset_b,
                      transform_gr_frag_b_to_lw_frag_b(gr_frag_b_val),
                      lds_ld);

    // Initialize accumulation fragments
    mma_frag_acc mma_frag_acc_val;
    fill_fragment(mma_frag_acc_val, 0.0f);

    // Synchronize workgroup
    synchronize_workgroup();

    // Main accumulation loop with prefetching
    for(uint32_t current_k = macro_tile_k; current_k < k; current_k += macro_tile_k)
    {
        // Load from LDS buffer
        lr_frag_a lr_frag_a_val;
        lr_frag_b lr_frag_b_val;
        load_matrix_sync(lr_frag_a_val, lds_ptr_lo + lds_read_offset_a, lds_ld);
        load_matrix_sync(lr_frag_b_val, lds_ptr_lo + lds_read_offset_b, lds_ld);

        // Prefetch next global data
        load_matrix_sync(gr_frag_a_val, a + global_read_offset_a, lda);
        load_matrix_sync(gr_frag_b_val, b + global_read_offset_b, ldb);

        global_read_offset_a += k_step_offset_a;
        global_read_offset_b += k_step_offset_b;

        // Matrix multiply-accumulate
        mma_sync(mma_frag_acc_val,
                 transform_lr_frag_a_to_mma_frag_a(lr_frag_a_val),
                 transform_lr_frag_b_to_mma_frag_b(lr_frag_b_val),
                 mma_frag_acc_val);

        // Store prefetch to alternate LDS buffer
        store_matrix_sync(lds_ptr_hi + lds_write_offset_a,
                          transform_gr_frag_a_to_lw_frag_a(gr_frag_a_val),
                          lds_ld);
        store_matrix_sync(lds_ptr_hi + lds_write_offset_b,
                          transform_gr_frag_b_to_lw_frag_b(gr_frag_b_val),
                          lds_ld);

        synchronize_workgroup();

        // Swap LDS buffers
        auto* tmp  = lds_ptr_lo;
        lds_ptr_lo = lds_ptr_hi;
        lds_ptr_hi = tmp;
    }

    // Load C matrix
    using mma_frag_c_map_1d = GetDataLayout_t<mma_frag_c>;
    using mma_frag_d_map_1d = GetDataLayout_t<mma_frag_d>;

    mma_frag_c mma_frag_c_val;
    load_matrix_sync(mma_frag_c_val,
                     c + mma_frag_c_map_1d::fromMatrixCoord(warp_tile_coord, ldc),
                     ldc);

    // Final accumulation step
    lr_frag_a lr_frag_a_val;
    lr_frag_b lr_frag_b_val;
    load_matrix_sync(lr_frag_a_val, lds_ptr_lo + lds_read_offset_a, lds_ld);
    load_matrix_sync(lr_frag_b_val, lds_ptr_lo + lds_read_offset_b, lds_ld);
    mma_sync(mma_frag_acc_val,
             transform_lr_frag_a_to_mma_frag_a(lr_frag_a_val),
             transform_lr_frag_b_to_mma_frag_b(lr_frag_b_val),
             mma_frag_acc_val);

    // D = alpha * accum + beta * C (chunked for register efficiency)
    mma_frag_d         mma_frag_d_val;
    constexpr uint32_t chunk_size = 8u;
    constexpr uint32_t chunks     = mma_frag_d_val.num_elements / chunk_size;
    constexpr uint32_t remain     = mma_frag_d_val.num_elements % chunk_size;

    auto uniform_fma_unroll_chunk = [&](uint32_t start_idx, uint32_t size)
    {
        for(uint32_t i = start_idx; i < start_idx + size; i++)
        {
            mma_frag_d_val.x[i] = static_cast<output_t>(
                alpha * mma_frag_acc_val.x[i] + beta * static_cast<compute_t>(mma_frag_c_val.x[i]));
        }
    };

    for(uint32_t c = 0u; c < chunks; c++)
    {
        uniform_fma_unroll_chunk(c * chunk_size, chunk_size);
    }
    uniform_fma_unroll_chunk(chunks * chunk_size, remain);

    // Store final result
    store_matrix_sync(d + mma_frag_d_map_1d::fromMatrixCoord(warp_tile_coord, ldd),
                      mma_frag_d_val,
                      ldd);
}

int main()
{
    // 1. Set up input data
    constexpr uint32_t m = 256;
    constexpr uint32_t n = 256;
    constexpr uint32_t k = 256;

    constexpr compute_t alpha = 1.0f;
    constexpr compute_t beta  = 1.0f;

    // Runtime parameter validation
    uint32_t h_tblock_x    = is_gfx9() ? gfx9_params::tblock_x : gfx11_params::tblock_x;
    uint32_t h_tblock_y    = is_gfx9() ? gfx9_params::tblock_y : gfx11_params::tblock_y;
    uint32_t h_blocks_m    = is_gfx9() ? gfx9_params::blocks_m : gfx11_params::blocks_m;
    uint32_t h_blocks_n    = is_gfx9() ? gfx9_params::blocks_n : gfx11_params::blocks_n;
    uint32_t h_rocwmma_m   = is_gfx9() ? gfx9_params::rocwmma_m : gfx11_params::rocwmma_m;
    uint32_t h_rocwmma_n   = is_gfx9() ? gfx9_params::rocwmma_n : gfx11_params::rocwmma_n;
    uint32_t h_rocwmma_k   = is_gfx9() ? gfx9_params::rocwmma_k : gfx11_params::rocwmma_k;
    uint32_t h_warp_tile_m = h_blocks_m * h_rocwmma_m;
    uint32_t h_warp_tile_n = h_blocks_n * h_rocwmma_n;

    auto warp_size_val = get_warp_size();
    auto macro_tile_size_val
        = make_coord2d(h_tblock_x / warp_size_val * h_warp_tile_m, h_tblock_y * h_warp_tile_n);

    // Validate architecture and matrix dimensions
    if((is_gfx11() || is_gfx12()) && (h_rocwmma_m != 16 || h_rocwmma_n != 16))
    {
        std::cout << "Unsupported block size for this architecture!" << std::endl;
        return 0;
    }

    if(is_gfx9() && ((h_rocwmma_m != h_rocwmma_n) || (h_rocwmma_m != 16 && h_rocwmma_m != 32)))
    {
        std::cout << "Unsupported block size for GFX9!" << std::endl;
        return 0;
    }

    if((m < get<0>(macro_tile_size_val) || n < get<1>(macro_tile_size_val) || k < h_rocwmma_k)
       || (m % h_rocwmma_m | n % h_rocwmma_n | k % h_rocwmma_k))
    {
        std::cout << "Error: Matrix dimensions not supported" << std::endl;
        return 0;
    }

    // Calculate leading dimensions
    int lda = std::is_same_v<data_layout_a, row_major> ? k : m;
    int ldb = std::is_same_v<data_layout_b, row_major> ? n : k;
    int ldc = std::is_same_v<data_layout_c, row_major> ? n : m;
    int ldd = ldc;

    std::cout << "rocWMMA Performance HGEMM Example" << std::endl;
    std::cout << "High-performance kernel with data reuse and latency hiding" << std::endl;
    std::cout << "Matrix dimensions: A(" << m << "x" << k << ") * B(" << k << "x" << n << ") + C("
              << m << "x" << n << ") = D(" << m << "x" << n << ")" << std::endl;

    // 2. Initialize host matrices
    std::vector<input_t>  matrix_a(m * k);
    std::vector<input_t>  matrix_b(k * n);
    std::vector<output_t> matrix_c(m * n);
    std::vector<output_t> matrix_d(m * n, std::numeric_limits<output_t>::signaling_NaN());

    fill_rand(matrix_a.data(), m, k);
    fill_rand(matrix_b.data(), k, n);
    fill_rand(matrix_c.data(), m, n);

    // 3. Allocate device memory and copy input data
    input_t*  d_a;
    input_t*  d_b;
    output_t* d_c;
    output_t* d_d;

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

    // 4. Configure kernel launch parameters
    auto block_dim = dim3(h_tblock_x, h_tblock_y);
    auto grid_dim  = dim3(rocwmma::ceil_div(m, get<0>(macro_tile_size_val)),
                         rocwmma::ceil_div(n, get<1>(macro_tile_size_val)));

    // Calculate LDS usage for double buffering
    int lds_usage = 2u * sizeof(input_t)
                    * (get<0>(macro_tile_size_val) + get<1>(macro_tile_size_val)) * h_rocwmma_k;

    std::cout << "Launching performance kernel with grid(" << grid_dim.x << "," << grid_dim.y
              << ") block(" << block_dim.x << "," << block_dim.y << ")" << std::endl;

    // 5. Launch rocWMMA kernel
    // Warm-up
    for(uint32_t i = 0; i < 5; ++i)
    {
        gemm_rocwmma_d<<<grid_dim, block_dim, lds_usage>>>(m,
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
        gemm_rocwmma_d<<<grid_dim, block_dim, lds_usage>>>(m,
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

    std::cout << std::left << std::setw(8) << h_tblock_x << std::setw(8) << h_tblock_y
              << std::setw(8) << h_blocks_m << std::setw(8) << h_blocks_n << std::setw(6)
              << h_rocwmma_m << std::setw(6) << h_rocwmma_n << std::setw(6) << h_rocwmma_k
              << std::setw(8) << m << std::setw(8) << n << std::setw(8) << k << std::setw(8)
              << alpha << std::setw(8) << lda << std::setw(8) << ldb << std::setw(8) << beta
              << std::setw(8) << ldc << std::setw(8) << ldd << std::setw(13) << elapsed_time_ms
              << std::setw(23) << gflops << std::setw(10) << tflops_per_sec << std::endl;

    // 6. Copy result back to host
    HIP_CHECK(hipMemcpy(matrix_d.data(), d_d, bytes_d, hipMemcpyDeviceToHost));

    // 7. Validate result (CPU reference calculation)
    std::cout << "Validating result with reference..." << std::endl;
    std::vector<output_t> matrix_d_ref(m * n, std::numeric_limits<output_t>::signaling_NaN());
    gemm_cpu_h<input_t, output_t, compute_t, data_layout_a, data_layout_b, data_layout_c>(
        m,
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

    auto res = compare_equal(matrix_d.data(), matrix_d_ref.data(), m * n);

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
