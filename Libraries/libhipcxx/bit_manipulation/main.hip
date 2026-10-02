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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Layout of the packed 32-bit key. Each field is described by the position of
// its least significant bit (start) and its number of bits (width):
//
//   31                   12 11       4 3     1   0
//  +-----------------------+----------+-------+---+
//  |       sequence        | channel  | type  | f |
//  +-----------------------+----------+-------+---+
//
// All functions used below take the start and width as run-time `int`
// arguments and require 0 <= start, 0 <= width and start + width <= 32 for a
// 32-bit value.

/// \brief A 1-bit flag, stored in bit 0.
struct FlagField {
  static constexpr int start = 0;
  static constexpr int width = 1;
};

/// \brief A 3-bit type identifier (0-7), stored in bits 1-3.
struct TypeField {
  static constexpr int start = 1;
  static constexpr int width = 3;
};

/// \brief An 8-bit channel number (0-255), stored in bits 4-11.
struct ChannelField {
  static constexpr int start = 4;
  static constexpr int width = 8;
};

/// \brief A 20-bit sequence number, stored in bits 12-31.
struct SequenceField {
  static constexpr int start = 12;
  static constexpr int width = 20;
};

/// \brief The unpacked field values of one key.
struct KeyFields {
  uint32_t flag;
  uint32_t type;
  uint32_t channel;
  uint32_t sequence;
};

/// \brief All values that are computed for one key.
struct KeyResult {
  uint32_t key;              // The packed key.
  KeyFields fields;          // The fields extracted back from the key.
  uint32_t channel_in_place; // The channel bits, left at their position.
  uint32_t reversed;         // hip::bit_reverse(key)
  uint32_t rotated;          // hip::std::rotl(key, 32 - sequence start)
  int popcount;              // hip::std::popcount(key)
  int leading_zeros;         // hip::std::countl_zero(key)
  int bit_width;             // hip::std::bit_width(key)
};

/// \brief Derives the field values for the key with the given index. Every
/// value is kept within the width of its field.
__host__ __device__ constexpr KeyFields make_fields(const uint32_t index) {
  return KeyFields{index % 2u, (index / 2u) % (1u << TypeField::width),
                   (index * 37u) % (1u << ChannelField::width),
                   (index * 1021u) % (1u << SequenceField::width)};
}

/// \brief Packs the fields into a single 32-bit key.
__host__ __device__ constexpr uint32_t pack_key(const KeyFields fields) {
  // hip::bitfield_insert(dest, source, start, width) returns `dest` with the
  // bits [start, start + width) replaced by the lowest `width` bits of
  // `source`. The other bits of `dest` are kept, so the fields can be inserted
  // one after another into the same key. Higher bits of `source` that do not
  // fit into the field are discarded.
  uint32_t key = 0;
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

/// \brief Unpacks the fields of a 32-bit key.
__host__ __device__ constexpr KeyFields unpack_key(const uint32_t key) {
  // hip::bitfield_extract(value, start, width) returns the bits
  // [start, start + width) of `value`, shifted down to bit 0. This is the
  // inverse of hip::bitfield_insert.
  return KeyFields{
      hip::bitfield_extract(key, FlagField::start, FlagField::width),
      hip::bitfield_extract(key, TypeField::start, TypeField::width),
      hip::bitfield_extract(key, ChannelField::start, ChannelField::width),
      hip::bitfield_extract(key, SequenceField::start, SequenceField::width)};
}

// All libhipcxx bit functions are constexpr, so the key layout can be checked
// at compile time on the host. hip::bitmask<T>(start, width) returns a value
// of type T with the bits [start, start + width) set. The type T is not
// deduced from the arguments and must be given explicitly.
static_assert(hip::bitmask<uint32_t>(ChannelField::start,
                                     ChannelField::width) == 0x00000FF0u,
              "The channel occupies bits 4-11");
static_assert(FlagField::width + TypeField::width + ChannelField::width +
                      SequenceField::width ==
                  32,
              "The field widths add up to 32 bits");
static_assert(
    (hip::bitmask<uint32_t>(FlagField::start, FlagField::width) |
     hip::bitmask<uint32_t>(TypeField::start, TypeField::width) |
     hip::bitmask<uint32_t>(ChannelField::start, ChannelField::width) |
     hip::bitmask<uint32_t>(SequenceField::start, SequenceField::width)) ==
        0xFFFFFFFFu,
    "Together with the widths check, the fields cover every bit exactly once");
static_assert(
    unpack_key(pack_key(KeyFields{1u, 5u, 0xABu, 0x12345u})).channel == 0xABu,
    "A packed field can be extracted again");

/// \brief Builds the key with the given index and applies the bit operations
/// to it. The kernel calls this function on the device, and the host calls it
/// again to compute the reference values.
__host__ __device__ KeyResult analyze_key(const uint32_t index) {
  KeyResult result{};

  // Pack the fields into a key and unpack them again.
  result.key = pack_key(make_fields(index));
  result.fields = unpack_key(result.key);

  // Masking with hip::bitmask isolates the channel bits but, unlike
  // hip::bitfield_extract, leaves them at their position in the key.
  result.channel_in_place =
      result.key &
      hip::bitmask<uint32_t>(ChannelField::start, ChannelField::width);

  // hip::bit_reverse mirrors the bit order: bit 0 becomes bit 31, bit 1
  // becomes bit 30, and so on. It is a libhipcxx extension, not part of the
  // C++ Standard. cuda/__bit/bit_reverse.h selects one of three
  // implementations: on the device it uses the __brev intrinsic
  // (__bit_reverse_device). On the host, and during constant evaluation, it
  // uses the compiler builtin __builtin_bitreverse32 if available
  // (__bit_reverse_builtin), and a portable shift-and-mask sequence otherwise
  // (__bit_reverse_generic).
  result.reversed = hip::bit_reverse(result.key);

  // hip::std::rotl rotates the bits to the left. Bits shifted out at the top
  // re-enter at the bottom, so no bits are lost. Rotating by
  // 32 - SequenceField::start moves the sequence number into the lowest bits
  // and the other fields above it, which changes the order of the fields.
  result.rotated = hip::std::rotl(result.key, 32 - SequenceField::start);

  // The counting operations of the C++20 Standard header <bit>, provided by
  // libhipcxx for host and device code:
  // - hip::std::popcount counts the bits set to 1.
  // - hip::std::countl_zero counts the consecutive 0 bits, starting at the
  //   most significant bit. For a key of 0 it returns 32.
  // - hip::std::bit_width returns the number of bits needed to represent the
  //   value, which is 32 - countl_zero(key) for a 32-bit value.
  result.popcount = hip::std::popcount(result.key);
  result.leading_zeros = hip::std::countl_zero(result.key);
  result.bit_width = hip::std::bit_width(result.key);

  return result;
}

/// \brief Each thread builds and analyzes the key for its global index.
__global__ void analyze_keys_kernel(KeyResult *d_results,
                                    const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    d_results[index] = analyze_key(index);
  }
}

/// \brief Formats a value as an 8-digit hexadecimal number.
std::string to_hex(const uint32_t value) {
  std::stringstream stream;
  stream << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
  return stream.str();
}

/// \brief Formats a key as a binary number, most significant bit first, with
/// an underscore between two fields.
std::string to_fields_binary(const uint32_t key) {
  std::string binary;
  for (int bit = 31; bit >= 0; --bit) {
    binary += hip::bitfield_extract(key, bit, 1) ? '1' : '0';
    if (bit == SequenceField::start || bit == ChannelField::start ||
        bit == TypeField::start) {
      binary += '_';
    }
  }
  return binary;
}

/// \brief Returns whether two sets of fields are equal.
bool fields_equal(const KeyFields &a, const KeyFields &b) {
  return a.flag == b.flag && a.type == b.type && a.channel == b.channel &&
         a.sequence == b.sequence;
}

/// \brief Returns whether two results are equal.
bool results_equal(const KeyResult &a, const KeyResult &b) {
  return a.key == b.key && fields_equal(a.fields, b.fields) &&
         a.channel_in_place == b.channel_in_place && a.reversed == b.reversed &&
         a.rotated == b.rotated && a.popcount == b.popcount &&
         a.leading_zeros == b.leading_zeros && a.bit_width == b.bit_width;
}

int main() {
  // Number of keys.
  constexpr unsigned int size = 1024;

  // Number of threads per block.
  constexpr unsigned int block_size = 256;

  // Number of blocks, rounded up so that every key gets a thread.
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  // Allocate the device memory for the results.
  KeyResult *d_results{};
  HIP_CHECK(hipMalloc(&d_results, sizeof(KeyResult) * size));

  // Build and analyze the keys on the device.
  analyze_keys_kernel<<<grid_size, block_size>>>(d_results, size);

  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

  // Copy the results back to the host and free the device memory.
  std::vector<KeyResult> h_results(size);
  HIP_CHECK(hipMemcpy(h_results.data(), d_results, sizeof(KeyResult) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_results));

  // Print the first keys so that the results can be checked by eye.
  constexpr unsigned int print_count = 8;

  std::cout << "Packing fields with hip::bitfield_insert (binary: "
               "sequence_channel_type_flag):\n"
            << "index flag type channel sequence key        binary\n";
  for (unsigned int i = 0; i < print_count; ++i) {
    const KeyResult &result = h_results[i];
    std::cout << std::setw(5) << i << std::setw(5) << result.fields.flag
              << std::setw(5) << result.fields.type << std::setw(8)
              << result.fields.channel << std::setw(9) << result.fields.sequence
              << ' ' << to_hex(result.key) << ' '
              << to_fields_binary(result.key) << '\n';
  }

  // The rotation amount used below (32 - SequenceField::start) is derived
  // from the field layout instead of hard-coded, so the header cannot drift
  // out of sync with the actual rotl() call in analyze_key().
  constexpr int rotation_amount = 32 - SequenceField::start;
  std::cout << "\nBit operations on the packed key:\n"
            << "index key        popcount countl_zero bit_width bit_reverse "
               "rotl(key,"
            << rotation_amount << ") channel_bits\n";
  for (unsigned int i = 0; i < print_count; ++i) {
    const KeyResult &result = h_results[i];
    std::cout << std::setw(5) << i << ' ' << to_hex(result.key) << std::setw(9)
              << result.popcount << std::setw(12) << result.leading_zeros
              << std::setw(10) << result.bit_width << ' '
              << to_hex(result.reversed) << "  " << to_hex(result.rotated)
              << "   " << to_hex(result.channel_in_place) << '\n';
  }
  std::cout << std::endl;

  // Validate the results. The fields extracted on the device must match the
  // fields that were packed, and every value must match the value computed on
  // the host. The host and the device run the same source code, but several
  // functions take a different implementation path on each side, for example
  // hip::bit_reverse (intrinsic versus builtin) and hip::std::rotl.
  int errors = 0;
  for (unsigned int i = 0; i < size; ++i) {
    const KeyResult expected = analyze_key(i);
    const bool round_trip_ok =
        fields_equal(h_results[i].fields, make_fields(i));
    if (!round_trip_ok || !results_equal(h_results[i], expected)) {
      ++errors;
    }
  }

  return report_validation_result(errors);
}
