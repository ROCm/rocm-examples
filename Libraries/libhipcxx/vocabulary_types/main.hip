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

// libhipcxx vocabulary types used in this example, and what each one buys
// over the alternative:
// - hip::std::tuple / hip::std::pair (<hip/std/tuple>, <hip/std/utility>)
//   are the Standard product types, usable on host and device. HIP built-in
//   vector types (float4, int2, ...) are additionally given a tuple
//   interface (tuple_size, tuple_element, get<N>), so generic, index-based
//   code written for hip::std::tuple also works on them, without a
//   hand-written .x/.y/.z/.w accessor for every type. This does NOT extend
//   to structured bindings: `auto [x, y, z, w] = some_float4;` is
//   unsupported with hipcc because of how HIP_vector_type is implemented (a
//   union member), so element access below always goes through
//   hip::std::get<N> explicitly.
// - hip::std::expected<T, E> (<hip/std/expected>) is a per-thread return
//   value that is either a T or an E. Device code cannot use exceptions, so
//   without expected a failure would need an out-parameter or a sentinel
//   value; every thread below instead returns one expected, and the host
//   tells success from failure with has_value().
// - hip::std::optional<T> (<hip/std/optional>) is a nullable value with
//   the same has_value()/value_or() interface as expected. It is used below
//   for a bonus term that only some threads compute.
#include <hip/std/expected>
#include <hip/std/optional>
#include <hip/std/tuple>
#include <hip/std/type_traits>
#include <hip/std/utility>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/// \brief The two ways a thread's computation can fail.
enum class ErrorCode {
  DivisionByZero,
  OutOfRange,
};

/// \brief Formats an ErrorCode for printing.
std::string to_string(const ErrorCode code) {
  switch (code) {
  case ErrorCode::DivisionByZero:
    return "DivisionByZero";
  case ErrorCode::OutOfRange:
    return "OutOfRange";
  }
  return "Unknown";
}

/// \brief Everything computed for one thread.
struct ThreadResult {
  hip::std::expected<float, ErrorCode> value; // The final result or error.
  hip::std::optional<float> bonus; // Set only for every fourth thread.
};

// float4 is treated as a 4-element tuple of float by libhipcxx: tuple_size
// and tuple_element are specialized for it, and hip::std::get<N> works on
// it, just like on hip::std::tuple. These static_asserts are checked at
// compile time on the host.
static_assert(hip::std::tuple_size<float4>::value == 4,
              "float4 is treated as a 4-element tuple");
static_assert(
    hip::std::is_same<hip::std::tuple_element<0, float4>::type, float>::value,
    "Every element of float4 has type float");

/// \brief Computes the result for the thread with the given global index.
/// \p config is the tuple built once on the host and passed unchanged into
/// the kernel launch; every thread reads it and combines it with its own
/// index. The kernel calls this function on the device, and the host calls
/// it again, with the same config, to compute the reference values.
__host__ __device__ ThreadResult
analyze_record(const hip::std::tuple<int, float, float4> &config,
               const unsigned int index) {
  ThreadResult result{};

  // Unpack the shared configuration. hip::std::get<N> works identically on
  // hip::std::tuple and, thanks to the specializations above, on float4.
  const int numerator = hip::std::get<0>(config);
  const float scale = hip::std::get<1>(config);
  const float4 weights = hip::std::get<2>(config);
  const float weight_sum =
      hip::std::get<0>(weights) + hip::std::get<1>(weights) +
      hip::std::get<2>(weights) + hip::std::get<3>(weights);

  // Deliberate failure path #1: thread 0 would divide by its own index.
  if (index == 0) {
    result.value = hip::std::expected<float, ErrorCode>(
        hip::std::unexpect, ErrorCode::DivisionByZero);
    return result;
  }

  const float divided =
      static_cast<float>(numerator) / static_cast<float>(index);
  const float scaled = divided * scale * weight_sum;

  // An optional intermediate value: only every fourth thread computes a
  // "precision bonus". hip::std::nullopt leaves it unset for the others,
  // and value_or(0.0f) below folds the missing case into the computation
  // without an if/else.
  result.bonus = (index % 4 == 0) ? hip::std::optional<float>(scaled * 0.1f)
                                  : hip::std::nullopt;
  const float final_value = scaled + result.bonus.value_or(0.0f);

  // Deliberate failure path #2: small indices divide 120 by almost nothing
  // and leave the valid range. The transition happens between index 12
  // (final_value == 55.0, still out of range) and index 13
  // (final_value ~= 46.15, back in range): a comfortable margin away from
  // range_limit, so the host/device floating-point rounding differences
  // handled by nearly_equal below cannot flip an OutOfRange/success outcome,
  // only the exact value within one outcome.
  constexpr float range_limit = 50.0f;
  if (final_value > range_limit || final_value < -range_limit) {
    result.value = hip::std::expected<float, ErrorCode>(hip::std::unexpect,
                                                        ErrorCode::OutOfRange);
  } else {
    result.value = hip::std::expected<float, ErrorCode>(final_value);
  }

  return result;
}

/// \brief Each thread analyzes the record for its own global index.
__global__ void
analyze_records_kernel(const hip::std::tuple<int, float, float4> config,
                       ThreadResult *d_results, const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_results[index] = analyze_record(config, index);
  }
}

/// \brief Compares two floats computed the same way on host and device.
/// HIP compiles device code with -ffp-contract=fast by default, so an
/// expression such as `scaled + bonus` (where bonus is itself `scaled *
/// 0.1f`) may be fused into a single FMA on the GPU while the host build
/// rounds the multiplication and the addition separately. The two results
/// can then differ in the last ulp even though both are correctly rounded
/// for their own evaluation order, so an exact == would report spurious
/// validation failures on correct hardware. A small relative tolerance
/// absorbs that without hiding a real mismatch.
bool nearly_equal(const float a, const float b) {
  return std::abs(a - b) <= 1e-5f * std::max(1.0f, std::abs(b));
}

/// \brief Formats a thread's expected value or error for printing.
std::string format_result(const hip::std::expected<float, ErrorCode> &value) {
  std::ostringstream stream;
  if (value.has_value()) {
    stream << std::fixed << std::setprecision(2) << *value;
  } else {
    stream << to_string(value.error());
  }
  return stream.str();
}

int main() {
  // Number of records analyzed, one per thread.
  constexpr unsigned int size = 256;

  // Number of threads per block.
  constexpr unsigned int block_size = 64;

  // Number of blocks, rounded up so that every record gets a thread.
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  // Build the shared configuration on the host: a numerator, a scale
  // factor, and a HIP float4 vector of per-channel weights.
  // hip::std::make_tuple deduces float4 as the third element type, and the
  // resulting tuple crosses into the kernel launch below like any other
  // trivially copyable argument.
  const hip::std::tuple<int, float, float4> config =
      hip::std::make_tuple(120, 0.5f, make_float4(1.0f, 2.0f, 3.0f, 4.0f));

  // Allocate the device memory for the results.
  ThreadResult *d_results{};
  HIP_CHECK(hipMalloc(&d_results, sizeof(ThreadResult) * size));

  // Analyze the records on the device.
  analyze_records_kernel<<<grid_size, block_size>>>(config, d_results, size);

  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

  // Copy the results back to the host and free the device memory.
  std::vector<ThreadResult> h_results(size);
  HIP_CHECK(hipMemcpy(h_results.data(), d_results, sizeof(ThreadResult) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_results));

  // Print the first records so that both outcomes -- a value and an error
  // -- can be checked by eye.
  constexpr unsigned int print_count = 16;
  std::cout << "Per-thread results (numerator=120, scale=0.5, weights="
               "(1,2,3,4)):\n"
            << "index result          bonus\n";
  for (unsigned int i = 0; i < print_count; ++i) {
    const ThreadResult &result = h_results[i];
    std::cout << std::setw(5) << i << ' ' << std::setw(15) << std::left
              << format_result(result.value) << std::right << ' '
              << (result.bonus.has_value() ? std::to_string(*result.bonus)
                                           : std::string("--"))
              << '\n';
  }
  std::cout << std::endl;

  // Validate the results. Every value is recomputed on the host with the
  // same analyze_record function and the same config, and compared with
  // what the device produced. The run is also checked to actually exercise
  // both failure paths and the success path, so the design described above
  // is verified rather than assumed.
  int errors = 0;
  unsigned int success_count = 0;
  unsigned int division_by_zero_count = 0;
  unsigned int out_of_range_count = 0;
  for (unsigned int i = 0; i < size; ++i) {
    const ThreadResult expected_result = analyze_record(config, i);
    const ThreadResult &actual_result = h_results[i];

    // Values and bonuses are compared with nearly_equal, not ==: see the
    // comment on nearly_equal for why an exact comparison is not safe here.
    // has_value() and error() are still compared exactly, since they are
    // not the product of a floating-point computation.
    const bool values_match =
        actual_result.value.has_value() == expected_result.value.has_value() &&
        (actual_result.value.has_value()
             ? nearly_equal(*actual_result.value, *expected_result.value)
             : actual_result.value.error() == expected_result.value.error());
    const bool bonus_matches =
        actual_result.bonus.has_value() == expected_result.bonus.has_value() &&
        (!actual_result.bonus.has_value() ||
         nearly_equal(*actual_result.bonus, *expected_result.bonus));
    if (!values_match || !bonus_matches) {
      ++errors;
    }

    if (actual_result.value.has_value()) {
      ++success_count;
    } else if (actual_result.value.error() == ErrorCode::DivisionByZero) {
      ++division_by_zero_count;
    } else {
      ++out_of_range_count;
    }
  }

  std::cout << success_count << " successes, " << division_by_zero_count
            << " division-by-zero errors, " << out_of_range_count
            << " out-of-range errors (of " << size << " threads)\n";

  // Exactly thread 0 divides by zero by design.
  if (division_by_zero_count != 1) {
    ++errors;
  }
  // At least one small index must leave the valid range, and at least one
  // larger index must stay inside it; otherwise the example would only show
  // one of the two branches.
  if (out_of_range_count == 0 || success_count == 0) {
    ++errors;
  }

  // A short hip::std::pair <-> host std::pair round trip. The conversions
  // are host-only (there is no host std::pair on the device), so this runs
  // only here in main, not in the kernel.
  const hip::std::pair<int, float> hip_pair{7, 3.5f};
  // hip::std::pair's conversion operator to ::std::pair.
  const std::pair<int, float> std_pair{hip_pair};
  // hip::std::pair's converting constructor from ::std::pair.
  const hip::std::pair<int, float> round_trip{std_pair};

  std::cout << "hip::std::pair<int, float>{" << hip_pair.first << ", "
            << hip_pair.second << "} -> std::pair -> hip::std::pair{"
            << round_trip.first << ", " << round_trip.second << '}'
            << std::endl;

  if (round_trip.first != hip_pair.first ||
      round_trip.second != hip_pair.second) {
    ++errors;
  }

  return report_validation_result(errors);
}
