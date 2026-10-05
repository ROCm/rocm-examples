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

// libhipcxx vocabulary types used in this example: hip::std::tuple,
// hip::std::expected, hip::std::optional, and the tuple interface on HIP
// vector types such as float4. Structured bindings on float4 are unsupported
// with hipcc (a union member implementation detail), so element access always
// goes through hip::std::get<N>.
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
#include <string>
#include <utility>
#include <vector>

enum class ErrorCode {
  DivisionByZero,
  OutOfRange,
};

struct ThreadResult {
  hip::std::expected<float, ErrorCode> value; // The final result or error.
  hip::std::optional<float> bonus; // Set only for every fourth thread.
};

// float4 is treated as a 4-element tuple of float: tuple_size and
// tuple_element are specialized for it, and hip::std::get<N> works on it.
static_assert(hip::std::tuple_size<float4>::value == 4,
              "float4 is treated as a 4-element tuple");
static_assert(
    hip::std::is_same<hip::std::tuple_element<0, float4>::type, float>::value,
    "Every element of float4 has type float");

// Computes the result for the thread with the given global index. The kernel
// calls this function on the device, and the host calls it again with the
// same config to compute the reference values.
__host__ __device__ ThreadResult
analyze_record(const hip::std::tuple<int, float, float4> &config,
               const unsigned int index) {
  ThreadResult result{};

  // Deliberate failure path #1: thread 0 would divide by its own index.
  if (index == 0) {
    result.value = hip::std::expected<float, ErrorCode>(
        hip::std::unexpect, ErrorCode::DivisionByZero);
    return result;
  }

  const int numerator = hip::std::get<0>(config);
  const float scale = hip::std::get<1>(config);
  const float4 weights = hip::std::get<2>(config);
  const float weight_sum =
      hip::std::get<0>(weights) + hip::std::get<1>(weights) +
      hip::std::get<2>(weights) + hip::std::get<3>(weights);

  const float divided =
      static_cast<float>(numerator) / static_cast<float>(index);
  const float scaled = divided * scale * weight_sum;

  // Only every fourth thread computes a "precision bonus"; value_or(0.0f)
  // below folds the missing case into the computation.
  result.bonus = (index % 4 == 0) ? hip::std::optional<float>(scaled * 0.1f)
                                  : hip::std::nullopt;
  const float final_value = scaled + result.bonus.value_or(0.0f);

  // Deliberate failure path #2: small indices divide 120 by almost nothing
  // and leave the valid range.
  constexpr float range_limit = 50.0f;
  if (final_value > range_limit || final_value < -range_limit) {
    result.value = hip::std::expected<float, ErrorCode>(hip::std::unexpect,
                                                        ErrorCode::OutOfRange);
  } else {
    result.value = hip::std::expected<float, ErrorCode>(final_value);
  }

  return result;
}

__global__ void
analyze_records_kernel(const hip::std::tuple<int, float, float4> config,
                       ThreadResult *d_results, const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_results[index] = analyze_record(config, index);
  }
}

// HIP compiles device code with -ffp-contract=fast by default, so a
// host/device FMA rounding difference makes an exact == unreliable here.
bool nearly_equal(const float a, const float b) {
  return std::abs(a - b) <= 1e-5f * std::max(1.0f, std::abs(b));
}

int main() {
  constexpr unsigned int size = 256;
  constexpr unsigned int block_size = 64;
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  const hip::std::tuple<int, float, float4> config =
      hip::std::make_tuple(120, 0.5f, make_float4(1.0f, 2.0f, 3.0f, 4.0f));

  ThreadResult *d_results{};
  HIP_CHECK(hipMalloc(&d_results, sizeof(ThreadResult) * size));

  analyze_records_kernel<<<grid_size, block_size>>>(config, d_results, size);
  HIP_CHECK(hipGetLastError());

  std::vector<ThreadResult> h_results(size);
  HIP_CHECK(hipMemcpy(h_results.data(), d_results, sizeof(ThreadResult) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_results));

  for (unsigned int i = 0; i < 5; ++i) {
    const ThreadResult &result = h_results[i];
    if (result.value.has_value()) {
      std::cout << "thread " << i << ": value " << std::fixed
                << std::setprecision(2) << *result.value << ", bonus: "
                << (result.bonus.has_value() ? std::to_string(*result.bonus)
                                             : std::string("--"))
                << '\n';
    } else {
      std::cout << "thread " << i << ": error "
                << (result.value.error() == ErrorCode::DivisionByZero
                        ? "DivisionByZero"
                        : "OutOfRange")
                << '\n';
    }
  }

  // Recompute the results on the host with the same analyze_record function
  // and compare with what the device produced.
  int errors = 0;
  for (unsigned int i = 0; i < size; ++i) {
    const ThreadResult expected_result = analyze_record(config, i);
    const ThreadResult &actual_result = h_results[i];

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
  }
  std::cout << "Validated all " << size << " threads.\n";

  // A short hip::std::pair <-> host std::pair round trip.
  const hip::std::pair<int, float> hip_pair{7, 3.5f};
  const std::pair<int, float> std_pair{hip_pair};
  const hip::std::pair<int, float> round_trip{std_pair};

  std::cout << "hip::std::pair{" << hip_pair.first << ", " << hip_pair.second
            << "} -> std::pair -> {" << round_trip.first << ", "
            << round_trip.second << "}\n";

  if (round_trip.first != hip_pair.first ||
      round_trip.second != hip_pair.second) {
    ++errors;
  }

  return report_validation_result(errors);
}
