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
#include <iomanip>
#include <iostream>
#include <new>
#include <vector>

// Number of input elements each block looks at. This is the "tile size"; the
// kernel gets its slice of the input with span::subspan, not pointer
// arithmetic on a raw (pointer, length) pair.
constexpr unsigned int tile_size = 16;

// Fixed capacity of the per-block inplace_vector that collects hits. Chosen
// smaller than tile_size on purpose: a tile can contain more hits than the
// container can hold, so the example can show what happens at that limit.
constexpr unsigned int max_hits_per_block = 8;

// Total number of input elements. Not a multiple of tile_size, so the last
// block's tile is partial.
constexpr unsigned int size = 100;

// Threads per block. Only thread 0 does any work in this example (see
// find_hits_kernel), so this value is not performance-critical here.
constexpr unsigned int block_size = 64;

// Number of blocks, rounded up so that every input element belongs to a
// tile.
constexpr unsigned int grid_size = ceiling_div(size, tile_size);

/// \brief The predicate that decides whether a value is a "hit". Host and
/// device code use the same function, so that generating the input and
/// interpreting the kernel's output agree on what counts as a hit.
__host__ __device__ constexpr bool is_hit(const int value) {
  return value % 2 == 0;
}

/// \brief The per-block result, written back to device memory. Only the
/// first hit_count entries of `values` are meaningful.
struct BlockHits {
  unsigned int hit_count;         // hits.size() for this block.
  unsigned int overflow_count;    // Hits that did not fit (try_emplace_back
                                  // returned nullptr).
  int values[max_hits_per_block]; // The stored hit values.
};

// The type of the per-block hit collector. Named here so the __shared__
// storage below can be declared with `alignas(HitVector)` / `sizeof(HitVector)`
// without repeating the template arguments.
using HitVector = hip::std::inplace_vector<int, max_hits_per_block>;

/// \brief Scans one tile of `all` per block and collects the values that
/// satisfy is_hit into a fixed-capacity hip::std::inplace_vector built in
/// shared memory (LDS).
__global__ void find_hits_kernel(hip::std::span<const int> all,
                                 BlockHits *d_block_hits) {
  // Slice this block's tile out of the full input with subspan, instead of
  // pointer arithmetic. Guard the last tile, which may be smaller than
  // tile_size because all.size() need not be a multiple of it.
  const size_t offset = static_cast<size_t>(blockIdx.x) * tile_size;
  const size_t remaining = all.size() - offset;
  const size_t count =
      remaining < tile_size ? remaining : static_cast<size_t>(tile_size);
  const hip::std::span<const int> tile = all.subspan(offset, count);

  // __shared__ variables cannot have dynamic initialization in HIP: clang
  // only accepts a __shared__ variable whose type has a trivial (empty)
  // default constructor, and otherwise rejects the declaration outright
  // ("initialization is not supported for __shared__ variables").
  // hip::std::inplace_vector<int, N> selects an internal "trivial"
  // specialization for the trivial element type int, but that
  // specialization's defaulted default constructor is still not *empty*:
  // its members have default initializers that zero the element storage
  // and the size counter, so it has to execute code, which disqualifies it
  // from being declared as a plain __shared__ variable.
  //
  // The fix, mirrored from libhipcxx's own test support
  // (.upstream-tests/test/support/cuda_space_selector.h), is to reserve
  // raw, correctly aligned __shared__ bytes and construct the
  // inplace_vector into them explicitly with placement new. This also
  // illustrates the point of inplace_vector rather well: it needs only
  // raw storage of the right size and alignment to come alive, not an
  // allocator.
  alignas(HitVector) __shared__ unsigned char hits_storage[sizeof(HitVector)];

  // Simplification for clarity: thread 0 alone constructs the vector into
  // that storage, scans the tile, accumulates hits, writes the block's
  // result, and destroys the vector again. Since no thread other than
  // thread 0 ever touches `hits_storage`, none of this needs a
  // __syncthreads(): a barrier is only required when one thread must see
  // memory that another thread wrote, and that never happens here. (A
  // parallel scan with an atomic index into a shared inplace_vector would
  // need one; that is not the point of this example.)
  if (threadIdx.x == 0) {
    HitVector *hits = ::new (hits_storage) HitVector();

    unsigned int overflow_count = 0;
    for (size_t i = 0; i < tile.size(); ++i) {
      if (is_hit(tile[i])) {
        // inplace_vector has a fixed capacity and no allocator: it can
        // never grow past max_hits_per_block. try_emplace_back reports
        // that by returning a null pointer instead of allocating,
        // throwing, or invoking undefined behavior. (Plain emplace_back
        // would call hip::std::terminate() here once hits is full: on
        // the device, exceptions are not available, so libhipcxx's
        // __throw_bad_alloc terminates instead of throwing -- see
        // <cuda/std/__new/bad_alloc.h>.) try_emplace_back is therefore
        // always safe to call, even when more values match the predicate
        // than fit.
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

    // int is trivially destructible, so this call is a no-op in practice,
    // but it is included for correctness and for the teaching point: an
    // object placement-new'd into raw storage must be destroyed
    // explicitly, since nothing does that implicitly for __shared__
    // memory.
    hits->~HitVector();
  }
}

/// \brief Recomputes, independently of the kernel and without using
/// hip::std::inplace_vector, which of data[begin, end) are hits, and how
/// many did not fit into a container of capacity max_hits_per_block. Used
/// only to validate the device results.
struct ExpectedHits {
  std::vector<int> values;
  unsigned int overflow_count = 0;
};

ExpectedHits compute_expected_hits(const std::vector<int> &data,
                                   const size_t begin, const size_t end) {
  ExpectedHits expected;
  for (size_t i = begin; i < end; ++i) {
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
  // Build the input on the host. The values are index-derived so the run
  // is reproducible, but the general formula alone would not reliably
  // exercise every interesting case, so two tiles are overwritten on
  // purpose below.
  std::vector<int> h_data(size);
  for (unsigned int i = 0; i < size; ++i) {
    h_data[i] = static_cast<int>((i * 48271u + 12345u) % 41) - 20;
  }
  // Block 0's tile is made entirely even, so every one of its tile_size
  // elements is a hit -- more than max_hits_per_block fit, which is
  // exactly the "at/beyond the limit" case this example is meant to show.
  for (unsigned int i = 0; i < tile_size; ++i) {
    h_data[i] = static_cast<int>(2 * i);
  }
  // Block 2's tile is made entirely odd, so it has zero hits and its
  // inplace_vector stays empty.
  for (unsigned int i = 2 * tile_size; i < 3 * tile_size; ++i) {
    h_data[i] = static_cast<int>(2 * i + 1);
  }

  // Copy the input to the device and wrap it in a span: the kernel
  // argument is a (pointer, length) pair bundled into one object, not a
  // raw pair passed as two separate parameters.
  int *d_data{};
  HIP_CHECK(hipMalloc(&d_data, sizeof(int) * size));
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), sizeof(int) * size,
                      hipMemcpyHostToDevice));
  const hip::std::span<const int> all(d_data, size);

  // Allocate device memory for one result per block.
  BlockHits *d_block_hits{};
  HIP_CHECK(hipMalloc(&d_block_hits, sizeof(BlockHits) * grid_size));

  find_hits_kernel<<<grid_size, block_size>>>(all, d_block_hits);
  HIP_CHECK(hipGetLastError());

  std::vector<BlockHits> h_block_hits(grid_size);
  HIP_CHECK(hipMemcpy(h_block_hits.data(), d_block_hits,
                      sizeof(BlockHits) * grid_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_block_hits));
  HIP_CHECK(hipFree(d_data));

  // Print and validate the per-block results.
  int errors = 0;
  unsigned int total_hits = 0;
  unsigned int total_overflow = 0;

  std::cout << "Per-block hits (tile_size=" << tile_size
            << ", inplace_vector capacity=" << max_hits_per_block << "):\n"
            << "block tile_range      hit_count overflow values\n";
  for (unsigned int block = 0; block < grid_size; ++block) {
    const size_t begin = static_cast<size_t>(block) * tile_size;
    const size_t end = (begin + tile_size < size) ? begin + tile_size : size;
    const ExpectedHits expected = compute_expected_hits(h_data, begin, end);
    const BlockHits &actual = h_block_hits[block];

    std::cout << std::setw(3) << block << "   [" << std::setw(3) << begin
              << ", " << std::setw(3) << end << ")" << std::setw(9)
              << actual.hit_count << std::setw(9) << actual.overflow_count
              << "   ";
    for (unsigned int i = 0; i < actual.hit_count; ++i) {
      std::cout << actual.values[i] << (i + 1 < actual.hit_count ? ", " : "");
    }
    std::cout << '\n';

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

  // A short, self-contained demonstration of hip::std::array with get<N>
  // and structured bindings, independent of the hit-finding kernel above.
  const hip::std::array<unsigned int, 3> summary = {total_hits, total_overflow,
                                                    grid_size};

  // hip::std::get<N> accesses one element of a fixed-size array by a
  // compile-time index, the same way hip::std::tuple does.
  std::cout << "\nhip::std::array summary (via hip::std::get):\n"
            << "  total hits:     " << hip::std::get<0>(summary) << '\n'
            << "  total overflow: " << hip::std::get<1>(summary) << '\n'
            << "  blocks:         " << hip::std::get<2>(summary) << std::endl;

  // A structured binding unpacks the same array by position, using the
  // same tuple-like protocol (tuple_size, tuple_element, get) that
  // hip::std::get relies on.
  const auto &[summary_hits, summary_overflow, summary_blocks] = summary;
  std::cout << "hip::std::array summary (via structured bindings):\n"
            << "  total hits:     " << summary_hits << '\n'
            << "  total overflow: " << summary_overflow << '\n'
            << "  blocks:         " << summary_blocks << std::endl;

  if (summary_hits != total_hits || summary_overflow != total_overflow ||
      summary_blocks != grid_size) {
    ++errors;
  }

  return report_validation_result(errors);
}
