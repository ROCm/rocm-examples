// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "example_utils.hpp"

// hip::thread_scope selects which set of threads an atomic operation must
// synchronize with: thread_scope_thread (the calling thread only),
// thread_scope_block (threads of the same HIP block), thread_scope_device
// (threads anywhere on the same GPU) or thread_scope_system (also threads on
// the host or on other GPUs). A companion example, atomic_counter, introduces
// the scope parameter with a single, device-scope atomic. This example shows
// why you would narrow the scope: a block-scope atomic that lives in LDS
// (shared memory) is much cheaper to update than a device-scope atomic,
// because it never has to leave the compute unit that owns the block. Scope
// selection is therefore both a correctness decision (which threads must see
// the update) and a performance decision (how far that update has to
// travel), and it has no analogue in host-only C++, where std::atomic is
// always visible to the whole process.
#include <hip/atomic>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

/// \brief Number of blocks in the grid.
constexpr unsigned int grid_size = 8;

/// \brief Number of threads per block.
constexpr unsigned int block_size = 256;

/// \brief The block-scope atomic that combines the values of one block.
/// Lives in LDS (shared memory) and is only ever touched by the threads of
/// the block that owns it.
using BlockScopeAtomic = hip::atomic<int, hip::thread_scope_block>;

/// \brief Derives a value for one thread that depends on both its block and
/// its thread index, so that neither index alone determines the maximum:
/// the value must vary visibly within a block and across blocks for the
/// staged, scope-narrowed reduction below to be meaningful. The host calls
/// this same function to compute the reference results.
__host__ __device__ constexpr int thread_value(const unsigned int block_idx,
                                               const unsigned int thread_idx) {
  const unsigned int mixed =
      (thread_idx + 1u) * (block_idx * 6151u + 2654435761u);
  return static_cast<int>(mixed % 100000u);
}

__global__ void block_reduce_max_kernel(int *d_device_max, int *d_block_max) {
  // __shared__ variables never run a constructor: store the identity
  // element explicitly.
  __shared__ BlockScopeAtomic block_max;

  if (threadIdx.x == 0) {
    block_max.store(INT_MIN, hip::memory_order_relaxed);
  }
  __syncthreads();

  const int value = thread_value(blockIdx.x, threadIdx.x);

  // Block-scope fetch_max: cheap, LDS-only traffic.
  block_max.fetch_max(value, hip::memory_order_relaxed);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int this_block_max = block_max.load(hip::memory_order_relaxed);
    d_block_max[blockIdx.x] = this_block_max;

    // Device-scope fetch_max: one per block, visible to the whole grid.
    hip::atomic_ref<int, hip::thread_scope_device> device_max(*d_device_max);
    device_max.fetch_max(this_block_max, hip::memory_order_relaxed);
  }
}

int main() {
  int *d_device_max{};
  HIP_CHECK(hipMalloc(&d_device_max, sizeof(int)));
  const int initial_max = INT_MIN;
  // Seed the accumulator with the identity element for max.
  HIP_CHECK(hipMemcpy(d_device_max, &initial_max, sizeof(int),
                      hipMemcpyHostToDevice));

  int *d_block_max{};
  HIP_CHECK(hipMalloc(&d_block_max, sizeof(int) * grid_size));

  block_reduce_max_kernel<<<grid_size, block_size>>>(d_device_max, d_block_max);
  HIP_CHECK(hipGetLastError());

  std::vector<int> h_block_max(grid_size);
  HIP_CHECK(hipMemcpy(h_block_max.data(), d_block_max, sizeof(int) * grid_size,
                      hipMemcpyDeviceToHost));
  int h_final_max{};
  HIP_CHECK(hipMemcpy(&h_final_max, d_device_max, sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_device_max));
  HIP_CHECK(hipFree(d_block_max));

  std::vector<int> ref_block_max(grid_size, INT_MIN);
  int ref_final_max = INT_MIN;
  for (unsigned int block_idx = 0; block_idx < grid_size; ++block_idx) {
    for (unsigned int thread_idx = 0; thread_idx < block_size; ++thread_idx) {
      const int value = thread_value(block_idx, thread_idx);
      ref_block_max[block_idx] = std::max(ref_block_max[block_idx], value);
    }
    ref_final_max = std::max(ref_final_max, ref_block_max[block_idx]);
  }

  std::cout << "Block maxima (block-scope atomic in LDS):\n";
  for (unsigned int block_idx = 0; block_idx < grid_size; ++block_idx) {
    std::cout << "  block " << block_idx << ": " << h_block_max[block_idx]
              << '\n';
  }
  std::cout << "Device maximum (atomic_ref, thread_scope_device): "
            << h_final_max << std::endl;

  int errors = 0;
  for (unsigned int block_idx = 0; block_idx < grid_size; ++block_idx) {
    if (h_block_max[block_idx] != ref_block_max[block_idx]) {
      ++errors;
    }
  }
  if (h_final_max != ref_final_max) {
    ++errors;
  }

  return report_validation_result(errors);
}
