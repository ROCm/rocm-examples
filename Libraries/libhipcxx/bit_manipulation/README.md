# libhipcxx Bit Manipulation Example

## Description

This example shows how to pack several fields into a single 32-bit key, extract them again, and inspect the key with bit counting operations. It combines two groups of libhipcxx functions, all of which you can call from both host and device code:

- The libhipcxx extensions from `<hip/bit>`: `hip::bitfield_insert`, `hip::bitfield_extract`, `hip::bitmask`, and `hip::bit_reverse`. These functions are not part of the C++ Standard.
- The C++20 Standard `<bit>` operations from `<hip/std/bit>`: `hip::std::popcount`, `hip::std::countl_zero`, `hip::std::bit_width`, and `hip::std::rotl`. libhipcxx provides them for host and device code, also in C++17 mode.

The key uses the following layout:

| Bits  | Field      | Width   |
|-------|------------|---------|
| 12-31 | `sequence` | 20 bits |
| 4-11  | `channel`  | 8 bits  |
| 1-3   | `type`     | 3 bits  |
| 0     | `flag`     | 1 bit   |

### Application flow

1. The key layout is checked at compile time with `static_assert`. Because the libhipcxx bit functions are `constexpr`, `hip::bitmask` verifies on the host that the fields cover all 32 bits without overlapping, and a packed field is extracted again at compile time.
2. Device memory is allocated for one result per key.
3. The `analyze_keys_kernel` kernel is launched with one thread per key. Each thread:
   1. Derives the four field values from its global index.
   2. Packs the fields into a key with `hip::bitfield_insert` and extracts them again with `hip::bitfield_extract`.
   3. Isolates the channel bits at their position in the key with `hip::bitmask`.
   4. Computes `hip::bit_reverse`, `hip::std::rotl`, `hip::std::popcount`, `hip::std::countl_zero`, and `hip::std::bit_width` of the key.
   5. Writes all values to device memory.
4. The results are copied back to the host and the device memory is freed.
5. Two tables with the results for the first eight keys are printed to the standard output. The first table shows the fields and the packed key in hexadecimal and binary notation, with an underscore between two fields. The second table shows the results of the bit operations.
6. The host checks that the extracted fields match the packed fields, and computes every value again with the same functions. The device results are compared with the host results, and the result of the comparison is printed to the standard output.

## Key APIs and Concepts

- `hip::bitfield_insert(dest, source, start, width)` returns `dest` with the bits `[start, start + width)` replaced by the lowest `width` bits of `source`. All other bits of `dest` are kept, so you can insert several fields into the same value one after another.
- `hip::bitfield_extract(value, start, width)` returns the bits `[start, start + width)` of `value`, shifted down to bit 0. It is the inverse of `hip::bitfield_insert`.
- `hip::bitmask<T>(start, width)` returns a value of type `T` with the bits `[start, start + width)` set. The template argument `T` is not deduced, so you must specify it. Masking a value with a bitmask isolates a field but, unlike `hip::bitfield_extract`, leaves the bits at their position.
- `hip::bit_reverse(value)` reverses the order of the bits: bit 0 becomes the most significant bit and vice versa. The implementation in `cuda/__bit/bit_reverse.h` uses the `__brev` intrinsic on the device, the compiler builtin `__builtin_bitreverse32` on the host and during constant evaluation, and a portable shift-and-mask implementation if the builtin is not available.
- `hip::std::rotl(value, count)` rotates the bits to the left. Bits that leave at the top re-enter at the bottom, so no bits are lost. In this example, rotating by `32 - 12 = 20` moves the `sequence` field to the lowest bits.
- `hip::std::popcount(value)` counts the bits set to 1. `hip::std::countl_zero(value)` counts the consecutive 0 bits, starting at the most significant bit. `hip::std::bit_width(value)` returns the number of bits needed to represent the value, which is `32 - countl_zero(value)` for a 32-bit value.
- All these functions accept only unsigned integer types and are `constexpr`. The `start` and `width` arguments must satisfy `start >= 0`, `width >= 0`, and `start + width <=` the number of bits of the type.

## Demonstrated API Calls

### libhipcxx

- `hip::bitfield_insert`
- `hip::bitfield_extract`
- `hip::bitmask`
- `hip::bit_reverse`
- `hip::std::popcount`
- `hip::std::countl_zero`
- `hip::std::bit_width`
- `hip::std::rotl`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
