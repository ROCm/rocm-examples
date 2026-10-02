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

// <hip/cmath> provides the libhipcxx index-math helpers hip::ceil_div,
// hip::round_up and hip::round_down, along with the C++ Standard <cmath>
// declarations (via <hip/std/cmath>). None of the three helpers are part of
// the C++ Standard; they are libhipcxx extensions.
#include <hip/cmath>

#include <hip/hip_runtime.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

/// \brief Number of elements to process. Deliberately not a multiple of
/// block_size, so the launch grid computed below overshoots problem_size and
/// the kernel needs a bounds check.
constexpr unsigned int problem_size = 1000;

/// \brief Threads per block.
constexpr unsigned int block_size = 256;

/// \brief Tile/alignment granularity used for the round_up / round_down
/// examples further down.
constexpr unsigned int tile_size = 128;

// All three helpers are constexpr, so the results used throughout this
// example can be, and are, checked at compile time.
static_assert(hip::ceil_div(problem_size, block_size) == 4,
              "hip::ceil_div can size a launch grid at compile time");
static_assert(hip::round_up(problem_size, tile_size) == 1024,
              "hip::round_up rounds problem_size up to the next multiple of "
              "tile_size");
static_assert(hip::round_down(300u, tile_size) == 256,
              "hip::round_down rounds an offset down to the start of its "
              "containing tile");
static_assert(hip::round_up(tile_size, tile_size) == tile_size,
              "hip::round_up leaves an already-aligned value unchanged");
static_assert(hip::round_down(0u, tile_size) == 0,
              "hip::round_down of zero is zero");

/// \brief Writes each thread's global index to d_out. The launch grid is
/// sized with hip::ceil_div and may therefore contain more threads than
/// problem_size, so out-of-range threads must do nothing.
__global__ void write_indices_kernel(unsigned int *d_out,
                                     const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_out[index] = index;
  }
}

int main() {
  int errors = 0;

  // ---------------------------------------------------------------------
  // 1. Grid sizing with hip::ceil_div.
  // ---------------------------------------------------------------------

  // hip::ceil_div(a, b) takes a different implementation path depending on
  // where it runs (include/cuda/__cmath/ceil_div.h), but only when a / b's
  // promoted type is unsigned - the common case for sizes, and the case
  // here. It then uses NV_IF_ELSE_TARGET to pick between:
  //   - device: min(a, 1 + (a - 1) / b)      -- avoids computing a remainder
  //   - host:   res = a / b; res + (res * b != a)
  // As the header comments put it: "the ::min method is faster even if b is
  // a compile-time constant". When a / b's promoted type is signed instead,
  // host and device use the very same (a + b - 1) / b-style formula; there
  // is no separate device path in that case. Both formulas sidestep the
  // overflow that the naive (a + b - 1) / b idiom is prone to; see part 2
  // below.
  const unsigned int grid_size = hip::ceil_div(problem_size, block_size);

  std::cout << "Grid sizing with hip::ceil_div:\n"
            << "  problem_size                        = " << problem_size
            << '\n'
            << "  block_size                          = " << block_size << '\n'
            << "  grid_size = ceil_div(problem_size, block_size) = "
            << grid_size << '\n'
            << "  threads launched (grid_size * block_size)      = "
            << grid_size * block_size << " (overshoots problem_size by "
            << grid_size * block_size - problem_size << ")\n"
            << std::endl;

  unsigned int *d_indices{};
  HIP_CHECK(hipMalloc(&d_indices, sizeof(unsigned int) * problem_size));

  write_indices_kernel<<<grid_size, block_size>>>(d_indices, problem_size);
  HIP_CHECK(hipGetLastError());

  std::vector<unsigned int> h_indices(problem_size);
  HIP_CHECK(hipMemcpy(h_indices.data(), d_indices,
                      sizeof(unsigned int) * problem_size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_indices));

  for (unsigned int i = 0; i < problem_size; ++i) {
    if (h_indices[i] != i) {
      ++errors;
    }
  }
  std::cout << "Kernel wrote " << problem_size
            << " indices; the bounds check against problem_size kept the "
            << grid_size * block_size - problem_size
            << " overshoot threads from writing out of range.\n"
            << std::endl;

  // ---------------------------------------------------------------------
  // 2. The overflow pitfall of the naive (a + b - 1) / b idiom.
  // ---------------------------------------------------------------------

  // With a = UINT_MAX and b = 2, the naive idiom computes
  // (UINT_MAX + 2 - 1) / 2 = (UINT_MAX + 1) / 2. UINT_MAX + 1 wraps around to
  // 0 in unsigned arithmetic, so the naive formula silently returns 0 instead
  // of the correct result. hip::ceil_div avoids this failure mode entirely,
  // which is the actual argument for using it instead of hand-rolling the
  // formula.
  constexpr unsigned int max_value = std::numeric_limits<unsigned int>::max();
  constexpr unsigned int naive_wrong_result = (max_value + 2u - 1u) / 2u;
  constexpr unsigned int correct_result = hip::ceil_div(max_value, 2u);
  static_assert(naive_wrong_result == 0u,
                "the naive (a + b - 1) / b idiom overflows silently for a "
                "dividend near the maximum value of its type");
  static_assert(correct_result == max_value / 2u + 1u,
                "hip::ceil_div computes the correct result without "
                "overflowing");

  std::cout << "Overflow pitfall of the naive (a + b - 1) / b idiom (a = "
               "UINT_MAX, b = 2):\n"
            << "  naive (a + b - 1) / b : " << naive_wrong_result
            << " (wrong; a + b - 1 wraps around)\n"
            << "  hip::ceil_div(a, b)  : " << correct_result << " (correct)\n"
            << std::endl;

  // ---------------------------------------------------------------------
  // 3. Tile/alignment sizing with hip::round_up and hip::round_down.
  // ---------------------------------------------------------------------

  // Round the problem size up to a multiple of the tile size before
  // allocating, so that the allocation covers only whole tiles.
  const unsigned int buffer_size = hip::round_up(problem_size, tile_size);
  std::cout << "Rounding a buffer size up to a tile boundary before "
               "allocating:\n"
            << "  round_up(" << problem_size << ", " << tile_size
            << ") = " << buffer_size << '\n'
            << std::endl;

  // Round an arbitrary offset down to the start of the tile that contains
  // it, e.g. to find where the tile covering that offset begins.
  constexpr unsigned int offset = 300;
  const unsigned int tile_start = hip::round_down(offset, tile_size);
  std::cout << "Rounding an offset down to the start of its containing "
               "tile:\n"
            << "  round_down(" << offset << ", " << tile_size
            << ") = " << tile_start << '\n'
            << std::endl;

  // Boundary cases: zero, one below a tile boundary, exactly on a boundary,
  // one above a boundary, and a couple of larger values (including the two
  // examples above, so they are also covered by the validation below).
  const std::vector<unsigned int> boundary_values = {
      0u,     1u,           tile_size - 1u, tile_size, tile_size + 1u,
      offset, problem_size, 4u * tile_size};

  std::cout << "Boundary cases for hip::round_up / hip::round_down "
               "(tile_size = "
            << tile_size << "):\n"
            << "     value   round_up round_down\n";
  for (const unsigned int value : boundary_values) {
    const unsigned int up = hip::round_up(value, tile_size);
    const unsigned int down = hip::round_down(value, tile_size);
    std::cout << std::setw(10) << value << std::setw(11) << up << std::setw(11)
              << down << '\n';

    // Independent host oracle for both results. This uses plain division and
    // a remainder check, not the naive (a + b - 1) / b idiom, so the check
    // does not share hip::ceil_div's implementation with what it verifies.
    const unsigned int expected_ceil_div =
        value / tile_size + (value % tile_size != 0 ? 1u : 0u);
    const unsigned int expected_up = expected_ceil_div * tile_size;
    const unsigned int expected_down = (value / tile_size) * tile_size;
    if (up != expected_up || down != expected_down) {
      ++errors;
    }
  }
  std::cout << std::endl;

  return report_validation_result(errors);
}
