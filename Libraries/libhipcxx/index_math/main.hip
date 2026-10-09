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

#include <iostream>
#include <limits>
#include <vector>

// Deliberately not a multiple of block_size, so the launch grid computed
// below overshoots problem_size and the kernel needs a bounds check.
constexpr unsigned int problem_size = 1000;
constexpr unsigned int block_size = 256;
constexpr unsigned int tile_size = 128;

// Writes each thread's global index to d_out. The launch grid is sized with
// hip::ceil_div and may therefore contain more threads than problem_size, so
// out-of-range threads must do nothing.
__global__ void write_indices_kernel(unsigned int *d_out,
                                     const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_out[index] = index;
  }
}

int main() {
  int errors = 0;

  // Grid sizing with hip::ceil_div.
  const unsigned int grid_size = hip::ceil_div(problem_size, block_size);
  std::cout << "hip::ceil_div(" << problem_size << ", " << block_size
            << ") = " << grid_size << " (" << grid_size * block_size
            << " threads launched)\n";

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
  std::cout << "Kernel wrote " << problem_size << " indices; the bounds "
            << "check kept the " << grid_size * block_size - problem_size
            << " overshoot threads from writing out of range.\n";

  // The overflow pitfall of the naive (a + b - 1) / b idiom: with
  // a = UINT_MAX and b = 2, a + b - 1 wraps around to 0.
  constexpr unsigned int max_value = std::numeric_limits<unsigned int>::max();
  constexpr unsigned int naive_wrong_result = (max_value + 2u - 1u) / 2u;
  constexpr unsigned int correct_result = hip::ceil_div(max_value, 2u);
  static_assert(naive_wrong_result == 0u,
                "the naive (a + b - 1) / b idiom overflows silently for a "
                "dividend near the maximum value of its type");
  static_assert(correct_result == max_value / 2u + 1u,
                "hip::ceil_div computes the correct result without "
                "overflowing");

  std::cout << "naive (a + b - 1) / b idiom (a = UINT_MAX, b = 2): "
            << naive_wrong_result << " (wrong)\n";
  std::cout << "hip::ceil_div(UINT_MAX, 2): " << correct_result
            << " (correct)\n";

  // Tile/alignment sizing with hip::round_up and hip::round_down.
  const unsigned int buffer_size = hip::round_up(problem_size, tile_size);
  std::cout << "hip::round_up(" << problem_size << ", " << tile_size
            << ") = " << buffer_size << '\n';

  constexpr unsigned int offset = 300;
  const unsigned int tile_start = hip::round_down(offset, tile_size);
  std::cout << "hip::round_down(" << offset << ", " << tile_size
            << ") = " << tile_start << '\n';

  // Validate round_up/round_down against a plain-division oracle.
  const std::vector<unsigned int> boundary_values = {
      0u, 1u, tile_size - 1u, tile_size, offset, problem_size};
  for (const unsigned int value : boundary_values) {
    const unsigned int up = hip::round_up(value, tile_size);
    const unsigned int down = hip::round_down(value, tile_size);
    std::cout << "value " << value << ": round_up = " << up
              << ", round_down = " << down << '\n';

    const unsigned int expected_ceil_div =
        value / tile_size + (value % tile_size != 0 ? 1u : 0u);
    if (up != expected_ceil_div * tile_size ||
        down != (value / tile_size) * tile_size) {
      ++errors;
    }
  }

  return report_validation_result(errors);
}
