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

// <cuda/mdspan> provides cuda::std::mdspan, the C++23 Standard non-owning
// multidimensional array view (usable in C++17 mode), together with the
// libhipcxx extensions cuda::restrict_mdspan and cuda::restrict_accessor. It
// already includes <cuda/std/mdspan> for the Standard pieces used below
// (cuda::std::extents, cuda::std::layout_right, cuda::std::layout_left, ...),
// so that header is not included again separately.
#include <cuda/mdspan>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

// The shape shared by every mdspan in this example: 8 rows and 12 columns,
// both fixed at compile time. Because every extent is static,
// Extents::rank_dynamic() == 0, so an Extents object can be built with the
// default constructor wherever an extents_type is expected -- including the
// `{}` shorthand from the libhipcxx documentation:
//   cuda::restrict_mdspan<int, cuda::std::extents<size_t, 4, 4>> md{ptr, {}};
// Here the `{}` list-initializes the second constructor argument, which
// selects mdspan's `mdspan(data_handle_type p, const extents_type& ext)`
// constructor and default-constructs an Extents value to fill in `ext`.
constexpr size_t num_rows = 8;
constexpr size_t num_cols = 12;

using Extents = cuda::std::extents<size_t, num_rows, num_cols>;

// One thread block handles one tile_size x tile_size tile of the matrix.
constexpr unsigned int tile_size = 4;

/// \brief Adds two matrices element-wise, `c(i, j) = a(i, j) + b(i, j)`.
///
/// The traditional signature for this kernel would be
/// `(const float *a, const float *b, float *c, size_t rows, size_t cols)`,
/// with `a[i * cols + j]` style indexing and a `__restrict__` keyword on
/// each pointer to tell the compiler that `a`, `b`, and `c` do not alias.
/// Here the kernel instead takes three mdspan values: the shape travels
/// with the data (`a.extent(0)`, `a.extent(1)`), elements are addressed
/// with `a(i, j)`, and the no-aliasing promise is expressed in the
/// *type* of the parameter rather than as a keyword on a raw pointer (see
/// below).
///
/// The function is a template so that the same source is instantiated once
/// with plain `cuda::std::mdspan` parameters and once with
/// `cuda::restrict_mdspan` parameters, without duplicating the kernel body.
template <class MDSpanA, class MDSpanB, class MDSpanC>
__global__ void add_matrices_kernel(MDSpanA a, MDSpanB b, MDSpanC c) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < static_cast<unsigned int>(a.extent(0)) &&
      col < static_cast<unsigned int>(a.extent(1))) {
    c(row, col) = a(row, col) + b(row, col);
  }
}

/// \brief Returns a value that depends on both the row and the column, so
/// that every matrix element is distinct and a reader can spot indexing
/// mistakes at a glance.
constexpr float matrix_value(const size_t i, const size_t j, const float base) {
  return base + static_cast<float>(i * 100 + j);
}

int main() {
  constexpr size_t size = num_rows * num_cols;

  // Host input matrices, stored row-major (the same layout that
  // cuda::std::layout_right, the default LayoutPolicy, will assume below).
  std::vector<float> h_a(size);
  std::vector<float> h_b(size);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      h_a[i * num_cols + j] = matrix_value(i, j, 0.0f);
      h_b[i * num_cols + j] = matrix_value(i, j, 1000.0f);
    }
  }

  float *d_a{};
  float *d_b{};
  float *d_c_plain{};
  float *d_c_restrict{};
  HIP_CHECK(hipMalloc(&d_a, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_b, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_c_plain, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_c_restrict, sizeof(float) * size));
  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  const dim3 block_dim(tile_size, tile_size);
  const dim3 grid_dim(
      ceiling_div(static_cast<unsigned int>(num_cols), tile_size),
      ceiling_div(static_cast<unsigned int>(num_rows), tile_size));

  // First instantiation of add_matrices_kernel: plain cuda::std::mdspan
  // parameters, using the default LayoutPolicy (cuda::std::layout_right) and
  // the default AccessorPolicy (cuda::std::default_accessor). Every mdspan is
  // built with the data-handle-and-extents constructor
  // `mdspan(data_handle_type p, const extents_type& ext)`, the same
  // constructor that cuda::restrict_mdspan uses below -- restrict_mdspan is a
  // libhipcxx alias for cuda::std::mdspan with a different AccessorPolicy,
  // not a separate class.
  using PlainMDSpanIn = cuda::std::mdspan<const float, Extents>;
  using PlainMDSpanOut = cuda::std::mdspan<float, Extents>;
  const PlainMDSpanIn a_plain(d_a, Extents{});
  const PlainMDSpanIn b_plain(d_b, Extents{});
  const PlainMDSpanOut c_plain(d_c_plain, Extents{});
  add_matrices_kernel<<<grid_dim, block_dim>>>(a_plain, b_plain, c_plain);
  HIP_CHECK(hipGetLastError());

  // Second instantiation of the very same function template, now with
  // cuda::restrict_mdspan parameters.
  // cuda::restrict_mdspan<T, Extents, Layout, Accessor> is an alias for
  // cuda::std::mdspan<T, Extents, Layout, cuda::restrict_accessor<Accessor>>.
  // cuda::restrict_accessor<Accessor> wraps Accessor and redeclares its
  // data_handle_type as `element_type * __restrict` (see
  // include/cuda/__mdspan/restrict_accessor.h); the accessor requires
  // Accessor::data_handle_type to already be a pointer. The upshot is that
  // "a, b, and c do not alias" becomes part of the *type* of the kernel
  // parameter instead of a __restrict__ keyword the mdspan-based signature
  // above has no room for, so the same aliasing information a hand-written
  // `(float *__restrict__, ...)` signature would give the compiler survives
  // the switch to mdspan parameters.
  using RestrictMDSpanIn = cuda::restrict_mdspan<const float, Extents>;
  using RestrictMDSpanOut = cuda::restrict_mdspan<float, Extents>;
  const RestrictMDSpanIn a_restrict(d_a, Extents{});
  const RestrictMDSpanIn b_restrict(d_b, Extents{});
  const RestrictMDSpanOut c_restrict(d_c_restrict, Extents{});
  add_matrices_kernel<<<grid_dim, block_dim>>>(a_restrict, b_restrict,
                                               c_restrict);
  HIP_CHECK(hipGetLastError());

  std::vector<float> h_c_plain(size);
  std::vector<float> h_c_restrict(size);
  HIP_CHECK(hipMemcpy(h_c_plain.data(), d_c_plain, sizeof(float) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_c_restrict.data(), d_c_restrict, sizeof(float) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c_plain));
  HIP_CHECK(hipFree(d_c_restrict));

  // Print a small slice of the result so that it can be checked by eye,
  // rather than the whole 8x12 matrix.
  constexpr size_t print_rows = 4;
  constexpr size_t print_cols = 6;
  std::cout << "c(i, j) = a(i, j) + b(i, j), top-left " << print_rows << "x"
            << print_cols << " corner (plain mdspan kernel):\n";
  for (size_t i = 0; i < print_rows; ++i) {
    for (size_t j = 0; j < print_cols; ++j) {
      std::cout << std::setw(9) << h_c_plain[i * num_cols + j];
    }
    std::cout << '\n';
  }
  std::cout << std::endl;

  int errors = 0;

  // The plain-mdspan kernel and the restrict_mdspan kernel must produce the
  // same numbers as a host-computed reference: cuda::restrict_mdspan only
  // adds an aliasing promise for the optimizer, it does not change what the
  // kernel computes.
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      const float expected =
          matrix_value(i, j, 0.0f) + matrix_value(i, j, 1000.0f);
      if (h_c_plain[i * num_cols + j] != expected) {
        ++errors;
      }
      if (h_c_restrict[i * num_cols + j] != expected) {
        ++errors;
      }
    }
  }

  // cuda::std::layout_right (the default LayoutPolicy used by every mdspan
  // above) is only half of the layout story. cuda::std::layout_left is a
  // drop-in replacement that changes how a multidimensional index is mapped
  // to a linear offset: layout_right is row-major (the *last* index varies
  // fastest, offset = i * num_cols + j), while layout_left is column-major
  // (the *first* index varies fastest, offset = i + j * num_rows). Building
  // an mdspan of each layout over the very same host buffer shows that the
  // same (i, j) then addresses two different elements of that buffer.
  const cuda::std::mdspan<const float, Extents, cuda::std::layout_right>
      right_view(h_a.data(), Extents{});
  const cuda::std::mdspan<const float, Extents, cuda::std::layout_left>
      left_view(h_a.data(), Extents{});

  std::cout << "\nlayout_right vs. layout_left over the same buffer "
               "h_a (index -> linear offset -> value):\n";
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const size_t right_offset = right_view.mapping()(i, j);
      const size_t left_offset = left_view.mapping()(i, j);
      std::cout << "(" << i << ", " << j << "): layout_right -> "
                << std::setw(2) << right_offset << " -> " << std::setw(6)
                << right_view(i, j) << "   layout_left -> " << std::setw(2)
                << left_offset << " -> " << std::setw(6) << left_view(i, j)
                << '\n';

      // layout_right: offset = i * num_cols + j (row-major).
      if (right_offset != i * num_cols + j) {
        ++errors;
      }
      // layout_left: offset = i + j * num_rows (column-major).
      if (left_offset != i + j * num_rows) {
        ++errors;
      }
    }
  }

  return report_validation_result(errors);
}
