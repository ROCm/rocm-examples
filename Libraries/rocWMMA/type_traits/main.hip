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

#include "rocwmma_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

// Five columns contain every residue modulo five, independent of rand() and
// its seed. Signed input must contain -3; unsigned input must retain +3.
template <typename T> bool check_fill(const char *name, bool signed_values) {
  std::array<T, 5> values{};
  fill_rand(values.data(), 1, values.size());
  std::array<float, 5> actual{};
  std::transform(values.begin(), values.end(), actual.begin(),
                 [](T value) { return static_cast<float>(value); });
  std::sort(actual.begin(), actual.end());
  const std::array<float, 5> expected =
      signed_values ? std::array<float, 5>{-3, 0, 1, 2, 4}
                    : std::array<float, 5>{0, 1, 2, 3, 4};
  if (actual != expected) {
    std::fprintf(stderr, "fill_rand generated incorrect signed values for %s\n",
                 name);
    return false;
  }
  return true;
}

int main() {
  bool ok = true;
  ok = check_fill<rocwmma::float8_t>("float8_t", true) && ok;
  ok = check_fill<rocwmma::bfloat8_t>("bfloat8_t", true) && ok;
  ok = check_fill<rocwmma::float8_fnuz_t>("float8_fnuz_t", true) && ok;
  ok = check_fill<rocwmma::bfloat8_fnuz_t>("bfloat8_fnuz_t", true) && ok;
  ok = check_fill<rocwmma::xfloat32_t>("xfloat32_t", true) && ok;
  ok = check_fill<float>("float", true) && ok;
  ok = check_fill<int>("int", true) && ok;
  ok = check_fill<unsigned int>("unsigned int", false) && ok;
  return ok ? 0 : 1;
}
