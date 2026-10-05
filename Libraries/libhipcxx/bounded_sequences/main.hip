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

// hip::std::span (a non-owning view over contiguous storage) and
// hip::std::inplace_vector (a fixed-capacity, allocator-free sequence
// container, backported here from C++26) are "bounded sequences": their
// storage requirements are known up front, so they work equally well in
// registers, in global memory, and -- as shown below -- in shared memory
// (LDS), where there is no allocator to call in the first place.
// hip::std::array is included too, for the small, self-contained demo of
// get<N> and structured bindings at the end of main().
#include <hip/std/array>
#include <hip/std/inplace_vector>
#include <hip/std/span>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <iostream>
#include <new>
#include <vector>

// Number of input elements each block looks at; the kernel gets its slice of
// the input with span::subspan.
constexpr unsigned int tile_size = 16;

// Fixed capacity of the per-block inplace_vector, deliberately smaller than
// tile_size so the example can show what happens when a tile holds more hits
// than fit.
constexpr unsigned int max_hits_per_block = 8;

// Total number of input elements; not a multiple of tile_size, so the last
// block's tile is partial.
constexpr unsigned int size = 100;

constexpr unsigned int block_size = 64;
constexpr unsigned int grid_size = ceiling_div(size, tile_size);

// The predicate that decides whether a value is a "hit"; host and device code
// use the same function.
__host__ __device__ constexpr bool is_hit(const int value) {
  return value % 2 == 0;
}

// The per-block result, written back to device memory. Only the first
// hit_count entries of `values` are meaningful.
struct BlockHits {
  unsigned int hit_count;
  unsigned int overflow_count; // Hits that did not fit.
  int values[max_hits_per_block];
};

// The type of the per-block hit collector.
using HitVector = hip::std::inplace_vector<int, max_hits_per_block>;

// Scans one tile of `all` per block and collects the hits into a
// fixed-capacity inplace_vector built in shared memory (LDS).
__global__ void find_hits_kernel(hip::std::span<const int> all,
                                 BlockHits *d_block_hits) {
  // Slice this block's tile out of the full input with subspan, guarding the
  // last tile, which may be smaller than tile_size.
  const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * tile_size;
  const std::size_t remaining = all.size() - offset;
  const std::size_t count =
      remaining < tile_size ? remaining : static_cast<std::size_t>(tile_size);
  const hip::std::span<const int> tile = all.subspan(offset, count);

  // __shared__ variables cannot have dynamic initialization, and
  // inplace_vector's default constructor is not trivial, so the vector has to
  // be constructed into raw __shared__ bytes with placement new.
  alignas(HitVector) __shared__ unsigned char hits_storage[sizeof(HitVector)];

  // Thread 0 alone does the work, so no __syncthreads() is needed here.
  if (threadIdx.x == 0) {
    HitVector *hits = ::new (hits_storage) HitVector();

    unsigned int overflow_count = 0;
    for (std::size_t i = 0; i < tile.size(); ++i) {
      if (is_hit(tile[i])) {
        // try_emplace_back returns a null pointer instead of growing past the
        // fixed capacity, so calling it is always safe.
        int *const stored = hits->try_emplace_back(tile[i]);
        if (stored == nullptr) {
          ++overflow_count;
        }
      }
    }

    BlockHits &out = d_block_hits[blockIdx.x];
    out.hit_count = static_cast<unsigned int>(hits->size());
    out.overflow_count = overflow_count;
    for (unsigned int i = 0; i < out.hit_count; ++i) {
      out.values[i] = (*hits)[i];
    }

    // An object placement-new'd into raw storage must be destroyed
    // explicitly.
    hits->~HitVector();
  }
}

// Recomputes, independently of the kernel, which of data[begin, end) are hits
// and how many did not fit; used only to validate the device results.
struct ExpectedHits {
  std::vector<int> values;
  unsigned int overflow_count = 0;
};

ExpectedHits compute_expected_hits(const std::vector<int> &data,
                                   const std::size_t begin,
                                   const std::size_t end) {
  ExpectedHits expected;
  for (std::size_t i = begin; i < end; ++i) {
    if (is_hit(data[i])) {
      if (expected.values.size() < max_hits_per_block) {
        expected.values.push_back(data[i]);
      } else {
        ++expected.overflow_count;
      }
    }
  }
  return expected;
}

int main() {
  // Index-derived input, so the run is reproducible.
  std::vector<int> h_data(size);
  for (unsigned int i = 0; i < size; ++i) {
    h_data[i] = static_cast<int>((i * 48271u + 12345u) % 41) - 20;
  }
  // Block 0's tile is made entirely even: more hits than fit, the
  // "beyond the limit" case.
  for (unsigned int i = 0; i < tile_size; ++i) {
    h_data[i] = static_cast<int>(2 * i);
  }
  // Block 2's tile is made entirely odd: zero hits.
  for (unsigned int i = 2 * tile_size; i < 3 * tile_size; ++i) {
    h_data[i] = static_cast<int>(2 * i + 1);
  }

  int *d_data{};
  HIP_CHECK(hipMalloc(&d_data, sizeof(int) * size));
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), sizeof(int) * size,
                      hipMemcpyHostToDevice));
  const hip::std::span<const int> all(d_data, size);

  BlockHits *d_block_hits{};
  HIP_CHECK(hipMalloc(&d_block_hits, sizeof(BlockHits) * grid_size));

  find_hits_kernel<<<grid_size, block_size>>>(all, d_block_hits);
  HIP_CHECK(hipGetLastError());

  std::vector<BlockHits> h_block_hits(grid_size);
  HIP_CHECK(hipMemcpy(h_block_hits.data(), d_block_hits,
                      sizeof(BlockHits) * grid_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_block_hits));
  HIP_CHECK(hipFree(d_data));

  // Validate the per-block results against the host oracle.
  int errors = 0;
  unsigned int total_hits = 0;
  unsigned int total_overflow = 0;
  for (unsigned int block = 0; block < grid_size; ++block) {
    const std::size_t begin = static_cast<std::size_t>(block) * tile_size;
    const std::size_t end =
        (begin + tile_size < size) ? begin + tile_size : size;
    const ExpectedHits expected = compute_expected_hits(h_data, begin, end);
    const BlockHits &actual = h_block_hits[block];

    bool block_ok = actual.hit_count == expected.values.size() &&
                    actual.overflow_count == expected.overflow_count;
    for (unsigned int i = 0; block_ok && i < actual.hit_count; ++i) {
      block_ok = actual.values[i] == expected.values[i];
    }
    if (!block_ok) {
      ++errors;
    }

    total_hits += actual.hit_count;
    total_overflow += actual.overflow_count;
  }
  std::cout << "Total hits: " << total_hits << ", total overflow: "
            << total_overflow << " across " << grid_size << " blocks\n";

  // A short, self-contained demonstration of hip::std::array with get<N> and
  // structured bindings.
  const hip::std::array<unsigned int, 3> summary = {total_hits, total_overflow,
                                                    grid_size};
  std::cout << "Summary via hip::std::get: hits "
            << hip::std::get<0>(summary) << ", overflow "
            << hip::std::get<1>(summary) << ", blocks "
            << hip::std::get<2>(summary) << '\n';

  const auto &[summary_hits, summary_overflow, summary_blocks] = summary;
  std::cout << "Summary via structured bindings: hits " << summary_hits
            << ", overflow " << summary_overflow << ", blocks "
            << summary_blocks << std::endl;

  if (summary_hits != total_hits || summary_overflow != total_overflow ||
      summary_blocks != grid_size) {
    ++errors;
  }

  return report_validation_result(errors);
}
