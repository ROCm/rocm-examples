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

// <hip/bit> provides the libhipcxx extensions hip::bitmask,
// hip::bit_reverse, hip::bitfield_insert and hip::bitfield_extract. It also
// includes <hip/std/bit>, which provides the C++20 Standard <bit> operations
// (hip::std::popcount, hip::std::countl_zero, hip::std::bit_width,
// hip::std::rotl, ...) for host and device code, even in C++17 mode.
#include <hip/bit>
#include <hip/std/bit>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

// Layout of the packed 32-bit key. Each field is described by the position of
// its least significant bit (start) and its number of bits (width):
//
//   31                   12 11       4 3     1   0
//  +-----------------------+----------+-------+---+
//  |       sequence        | channel  | type  | f |
//  +-----------------------+----------+-------+---+

// A 1-bit flag, stored in bit 0.
struct FlagField {
  static constexpr int start = 0;
  static constexpr int width = 1;
};

// A 3-bit type identifier (0-7), stored in bits 1-3.
struct TypeField {
  static constexpr int start = 1;
  static constexpr int width = 3;
};

// An 8-bit channel number (0-255), stored in bits 4-11.
struct ChannelField {
  static constexpr int start = 4;
  static constexpr int width = 8;
};

// A 20-bit sequence number, stored in bits 12-31.
struct SequenceField {
  static constexpr int start = 12;
  static constexpr int width = 20;
};

// The unpacked field values of one key.
struct KeyFields {
  std::uint32_t flag;
  std::uint32_t type;
  std::uint32_t channel;
  std::uint32_t sequence;
};

// All values that are computed for one key.
struct KeyResult {
  std::uint32_t key;
  KeyFields fields;
  std::uint32_t channel_in_place;
  std::uint32_t reversed;
  std::uint32_t rotated;
  int popcount;
  int leading_zeros;
  int bit_width;
};

// Derives the field values for the key with the given index; every value fits
// the width of its field.
__host__ __device__ constexpr KeyFields make_fields(const std::uint32_t index) {
  return KeyFields{index % 2u, (index / 2u) % (1u << TypeField::width),
                   (index * 37u) % (1u << ChannelField::width),
                   (index * 1021u) % (1u << SequenceField::width)};
}

// Packs the fields into a single 32-bit key. hip::bitfield_insert(dest,
// source, start, width) replaces the bits [start, start + width) of `dest`
// with the low `width` bits of `source`.
__host__ __device__ constexpr std::uint32_t pack_key(const KeyFields fields) {
  std::uint32_t key = 0;
  key = hip::bitfield_insert(key, fields.flag, FlagField::start,
                             FlagField::width);
  key = hip::bitfield_insert(key, fields.type, TypeField::start,
                             TypeField::width);
  key = hip::bitfield_insert(key, fields.channel, ChannelField::start,
                             ChannelField::width);
  key = hip::bitfield_insert(key, fields.sequence, SequenceField::start,
                             SequenceField::width);
  return key;
}

// Unpacks the fields of a 32-bit key. hip::bitfield_extract(value, start,
// width) returns the bits [start, start + width) of `value`, shifted down to
// bit 0.
__host__ __device__ constexpr KeyFields unpack_key(const std::uint32_t key) {
  return KeyFields{
      hip::bitfield_extract(key, FlagField::start, FlagField::width),
      hip::bitfield_extract(key, TypeField::start, TypeField::width),
      hip::bitfield_extract(key, ChannelField::start, ChannelField::width),
      hip::bitfield_extract(key, SequenceField::start, SequenceField::width)};
}

// All libhipcxx bit functions are constexpr, so the key layout can be checked
// at compile time; hip::bitmask<T>(start, width) needs T given explicitly.
static_assert(hip::bitmask<std::uint32_t>(ChannelField::start,
                                          ChannelField::width) == 0x00000FF0u,
              "The channel occupies bits 4-11");
static_assert(FlagField::width + TypeField::width + ChannelField::width +
                      SequenceField::width ==
                  32,
              "The field widths add up to 32 bits");
static_assert(
    unpack_key(pack_key(KeyFields{1u, 5u, 0xABu, 0x12345u})).channel == 0xABu,
    "A packed field can be extracted again");

// Builds the key with the given index and applies the bit operations to it.
// The kernel calls this on the device, and the host calls it again to compute
// the reference values.
__host__ __device__ KeyResult analyze_key(const std::uint32_t index) {
  KeyResult result{};

  result.key = pack_key(make_fields(index));
  result.fields = unpack_key(result.key);

  // Unlike hip::bitfield_extract, masking with hip::bitmask leaves the channel
  // bits at their position in the key.
  result.channel_in_place =
      result.key &
      hip::bitmask<std::uint32_t>(ChannelField::start, ChannelField::width);

  result.reversed = hip::bit_reverse(result.key);

  // Rotating by 32 - SequenceField::start moves the sequence number into the
  // lowest bits.
  result.rotated = hip::std::rotl(result.key, 32 - SequenceField::start);

  result.popcount = hip::std::popcount(result.key);
  result.leading_zeros = hip::std::countl_zero(result.key);
  result.bit_width = hip::std::bit_width(result.key);

  return result;
}

// Each thread builds and analyzes the key for its global index.
__global__ void analyze_keys_kernel(KeyResult *d_results,
                                    const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_results[index] = analyze_key(index);
  }
}

// Returns whether two results are equal.
bool results_equal(const KeyResult &a, const KeyResult &b) {
  return a.key == b.key && a.fields.flag == b.fields.flag &&
         a.fields.type == b.fields.type &&
         a.fields.channel == b.fields.channel &&
         a.fields.sequence == b.fields.sequence &&
         a.channel_in_place == b.channel_in_place && a.reversed == b.reversed &&
         a.rotated == b.rotated && a.popcount == b.popcount &&
         a.leading_zeros == b.leading_zeros && a.bit_width == b.bit_width;
}

int main() {
  constexpr unsigned int size = 1024;
  constexpr unsigned int block_size = 256;
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  KeyResult *d_results{};
  HIP_CHECK(hipMalloc(&d_results, sizeof(KeyResult) * size));

  analyze_keys_kernel<<<grid_size, block_size>>>(d_results, size);
  HIP_CHECK(hipGetLastError());

  std::vector<KeyResult> h_results(size);
  HIP_CHECK(hipMemcpy(h_results.data(), d_results, sizeof(KeyResult) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_results));

  // Print the first keys so that the results can be checked by eye.
  constexpr unsigned int print_count = 4;
  for (unsigned int i = 0; i < print_count; ++i) {
    const KeyResult &result = h_results[i];
    std::cout << "key " << i << ": 0x" << std::hex << result.key << std::dec
              << ", bit_reverse 0x" << std::hex << result.reversed << std::dec
              << ", rotl 0x" << std::hex << result.rotated << std::dec
              << ", channel bits 0x" << std::hex << result.channel_in_place
              << std::dec << ", popcount " << result.popcount
              << ", countl_zero " << result.leading_zeros << ", bit_width "
              << result.bit_width << '\n';
  }

  // The host and the device run the same source code, but some functions take
  // a different implementation path on each side.
  int errors = 0;
  for (unsigned int i = 0; i < size; ++i) {
    const KeyResult expected = analyze_key(i);
    if (!results_equal(h_results[i], expected)) {
      ++errors;
    }
  }

  return report_validation_result(errors);
}
