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

// <hip/stream_ref> provides hip::stream_ref, a small non-owning wrapper
// around a hipStream_t. In libhipcxx it is genuinely backed by hipStream_t
// (include/cuda/stream_ref, "using value_type = ::hipStream_t"), not a
// renamed CUDA type, so it drops into existing HIP code without a cast.
#include <hip/stream_ref>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

__host__ __device__ constexpr std::uint32_t
lcg_step(const std::uint32_t state) {
  return state * 1664525u + 1013904223u;
}

// Compiled for host and device so both sides run the exact same arithmetic:
// the kernel fills the buffer, and main() recomputes the reference values.
__host__ __device__ std::uint32_t
busy_fill_value(const std::uint32_t index, const unsigned int iterations) {
  std::uint32_t state = index;
  for (unsigned int i = 0; i < iterations; ++i) {
    state = lcg_step(state);
  }
  return state;
}

// The LCG loop gives the kernel a modest, non-zero amount of work, which is
// what makes the hip::stream_ref::ready() check meaningful.
__global__ void busy_fill_kernel(std::uint32_t *out, const unsigned int count,
                                 const unsigned int iterations) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = busy_fill_value(index, iterations);
  }
}

int main() {
  int errors = 0;

  constexpr unsigned int size = 1u << 16; // 65536
  constexpr unsigned int block_size = 256;
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  // Large enough that ready() can plausibly still observe a busy stream
  // right after the launch; small enough that host-side recomputation stays
  // fast.
  constexpr unsigned int busy_iterations = 256;

  // hip::stream_ref never owns the stream it wraps, so the hipStream_t below
  // is what actually owns the resource.
  hipStream_t raw;
  HIP_CHECK(hipStreamCreate(&raw));

  // Implicit conversion from hipStream_t: no cast or explicit wrapping needed.
  hip::stream_ref s{raw};

  // A default-constructed hip::stream_ref refers to the default stream.
  const hip::stream_ref default_ref;

  std::cout << "explicit stream handle: " << reinterpret_cast<void *>(s.get())
            << '\n';
  std::cout << "default stream handle: "
            << reinterpret_cast<void *>(default_ref.get()) << '\n';

  std::uint32_t *d_result{};
  HIP_CHECK(hipMalloc(&d_result, sizeof(std::uint32_t) * size));

  // The stream_ref must hand out the raw handle explicitly via get() for the
  // launch syntax.
  busy_fill_kernel<<<grid_size, block_size, 0, s.get()>>>(d_result, size,
                                                          busy_iterations);
  HIP_CHECK(hipGetLastError());

  // Racy, right-after-launch poll: false is the expected outcome but not
  // guaranteed, so it is only printed, not asserted.
  try {
    std::cout << "ready() right after launch: " << std::boolalpha
              << s.ready() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "ready() threw: " << e.what() << '\n';
    ++errors;
  }

  // Block until the kernel completes.
  try {
    s.wait();
  } catch (const std::exception &e) {
    std::cerr << "wait() threw: " << e.what() << '\n';
    ++errors;
  }

  // Not racy anymore: after wait(), ready() must report true.
  try {
    const bool ready = s.ready();
    std::cout << "ready() after wait(): " << std::boolalpha << ready << '\n';
    if (!ready) {
      ++errors;
    }
  } catch (const std::exception &e) {
    std::cerr << "ready() after wait() threw: " << e.what() << '\n';
    ++errors;
  }

  HIP_CHECK(hipStreamSynchronize(s.get()));

  std::vector<std::uint32_t> h_result(size);
  HIP_CHECK(hipMemcpy(h_result.data(), d_result, sizeof(std::uint32_t) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_result));

  for (unsigned int i = 0; i < 8; ++i) {
    std::cout << "h_result[" << i << "] = " << h_result[i] << '\n';
  }

  // hip::stream_ref compares by the wrapped handle's value, also directly
  // against a raw hipStream_t.
  const hip::stream_ref s_alias{raw};
  const bool same_stream_equal = (s == s_alias);
  const bool raw_conversion_equal = (s == raw);
  const bool default_stream_distinct = (s != default_ref);

  std::cout << std::boolalpha << "s == s_alias: " << same_stream_equal
            << ", s == raw: " << raw_conversion_equal
            << ", s != default_ref: " << default_stream_distinct << '\n';

  if (!same_stream_equal) {
    ++errors;
  }
  if (!raw_conversion_equal) {
    ++errors;
  }
  if (!default_stream_distinct) {
    ++errors;
  }

  // Validate the filled buffer against the host-computed reference.
  for (unsigned int i = 0; i < size; ++i) {
    if (h_result[i] != busy_fill_value(i, busy_iterations)) {
      ++errors;
    }
  }

  // hip::stream_ref is a non-owning view; the raw handle is needed for the
  // HIP call that releases the resource.
  HIP_CHECK(hipStreamDestroy(s.get()));

  return report_validation_result(errors);
}
