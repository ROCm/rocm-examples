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
//
// Note: the hip::thread_scope enum (cuda/std/__atomic/scopes.h) defines
// exactly thread_scope_system, thread_scope_device, thread_scope_block and
// thread_scope_thread - there is no thread_scope_cluster, even though HIP
// itself supports thread block clusters on some GPUs, for example gfx1250.
#include <hip/atomic>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <iomanip>
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

/// \brief Computes thread_value for every thread and combines the results
/// into a single maximum, in two stages:
///
/// 1. Block stage: every thread in a block applies fetch_max to a
///    block-scope atomic that lives in LDS (a __shared__ variable). This
///    atomic is only ever touched by threads of this one block, so
///    thread_scope_block is sufficient, and the read-modify-write never has
///    to leave the compute unit.
/// 2. Device stage: thread 0 of each block applies fetch_max, once, to a
///    device-scope atomic in global memory. This atomic is shared by every
///    block in the grid, so it needs thread_scope_device; narrowing it to
///    thread_scope_block here would be incorrect, since other blocks do
///    observe it.
///
/// The device-scope atomic is a hip::atomic_ref over a plain int, so the
/// host can initialize and read it back with an ordinary hipMemcpy instead
/// of a helper kernel: it is the same accumulator, but its storage is not
/// owned by an atomic object.
///
/// Because only one thread per block reaches the device stage, the grid
/// performs grid_size device-scope atomic operations instead of
/// grid_size * block_size, which is the performance payoff for staging the
/// reduction instead of applying fetch_max to the device-scope atomic
/// directly from every thread.
__global__ void block_reduce_max_kernel(int *d_device_max, int *d_block_max) {
  // __shared__ variables have no dynamic initializer: the compiler never
  // runs a constructor for them, and shared memory is not zero-initialized
  // between kernel launches. hip::atomic's default constructor is
  // "= default", and every base class behind it only holds a plain,
  // trivially-constructible storage member, so the type as a whole is
  // trivial. That means the object is already alive as soon as the
  // variable's storage exists, and calling store() on it below (with no
  // placement-new) is well defined. What still must happen explicitly is
  // giving it a known *value*: exactly one thread stores the identity
  // element for max (INT_MIN), and every thread synchronizes before relying
  // on it.
  __shared__ BlockScopeAtomic block_max;

  if (threadIdx.x == 0) {
    block_max.store(INT_MIN, hip::memory_order_relaxed);
  }
  __syncthreads();

  const int value = thread_value(blockIdx.x, threadIdx.x);

  // Block-scope fetch_max: cheap, LDS-only traffic. Correct here because no
  // other block ever reads or writes this shared-memory atomic.
  block_max.fetch_max(value, hip::memory_order_relaxed);
  __syncthreads(); // Wait until every thread's fetch_max has completed.

  if (threadIdx.x == 0) {
    const int this_block_max = block_max.load(hip::memory_order_relaxed);
    d_block_max[blockIdx.x] = this_block_max;

    // Device-scope fetch_max: one per block, visible to the whole grid (and,
    // with thread_scope_system, to the host and other GPUs too). The
    // atomic_ref layers atomic access over the plain int in d_device_max,
    // which the host can read back directly.
    hip::atomic_ref<int, hip::thread_scope_device> device_max(*d_device_max);
    device_max.fetch_max(this_block_max, hip::memory_order_relaxed);
  }
}

int main() {
  // The device-scope accumulator is a plain int, accessed atomically only
  // through a hip::atomic_ref inside the kernel. Allocate the accumulator,
  // the per-block maxima, and seed the accumulator with the identity element
  // for max; since it is an ordinary int, a host-to-device hipMemcpy can
  // initialize it without a helper kernel.
  int *d_device_max{};
  HIP_CHECK(hipMalloc(&d_device_max, sizeof(int)));
  const int initial_max = INT_MIN;
  HIP_CHECK(hipMemcpy(d_device_max, &initial_max, sizeof(int),
                      hipMemcpyHostToDevice));

  int *d_block_max{};
  HIP_CHECK(hipMalloc(&d_block_max, sizeof(int) * grid_size));

  // Run the staged, scope-narrowed reduction.
  block_reduce_max_kernel<<<grid_size, block_size>>>(d_device_max, d_block_max);
  HIP_CHECK(hipGetLastError());

  // Copy both the per-block maxima and the device-scope accumulator back to
  // the host. The accumulator is a plain int, so no load kernel is needed.
  std::vector<int> h_block_max(grid_size);
  HIP_CHECK(hipMemcpy(h_block_max.data(), d_block_max, sizeof(int) * grid_size,
                      hipMemcpyDeviceToHost));
  int h_final_max{};
  HIP_CHECK(hipMemcpy(&h_final_max, d_device_max, sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_device_max));
  HIP_CHECK(hipFree(d_block_max));

  // Independently compute the reference maxima on the host, using the same
  // thread_value function, so that a reader can check the printed values by
  // eye and so that the result can be validated.
  std::vector<int> ref_block_max(grid_size, INT_MIN);
  int ref_final_max = INT_MIN;
  for (unsigned int block_idx = 0; block_idx < grid_size; ++block_idx) {
    for (unsigned int thread_idx = 0; thread_idx < block_size; ++thread_idx) {
      const int value = thread_value(block_idx, thread_idx);
      ref_block_max[block_idx] = std::max(ref_block_max[block_idx], value);
    }
    ref_final_max = std::max(ref_final_max, ref_block_max[block_idx]);
  }

  // Print the block-scope partial maxima, one per block, so that a reader
  // can check that they vary across blocks.
  std::cout << "Block-scope reduction (hip::atomic<int, "
               "hip::thread_scope_block> in LDS):\n"
            << "block block_max reference match\n";
  for (unsigned int block_idx = 0; block_idx < grid_size; ++block_idx) {
    const bool match = h_block_max[block_idx] == ref_block_max[block_idx];
    std::cout << std::setw(5) << block_idx << std::setw(10)
              << h_block_max[block_idx] << std::setw(10)
              << ref_block_max[block_idx] << std::setw(8)
              << (match ? "yes" : "no") << '\n';
  }

  // Print the device-scope final maximum, combined from every block.
  std::cout << "\nDevice-scope reduction (hip::atomic_ref<int, "
               "hip::thread_scope_device>), combining "
            << grid_size << " block maxima:\n"
            << "device_max reference match\n"
            << std::setw(10) << h_final_max << std::setw(10) << ref_final_max
            << std::setw(8) << (h_final_max == ref_final_max ? "yes" : "no")
            << std::endl;

  // Validate: every block-scope maximum, and the final device-scope maximum,
  // must match the independently computed reference.
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
