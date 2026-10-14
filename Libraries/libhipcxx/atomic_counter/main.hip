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

// <hip/atomic> provides hip::atomic and hip::atomic_ref, the libhipcxx
// extensions that add an explicit thread scope to the C++ Standard atomic
// types. It also includes <hip/std/atomic>, which provides hip::std::atomic
// and hip::std::atomic_ref: the conforming Standard atomics, usable in both
// __host__ and __device__ code.
#include <hip/atomic>

#include <hip/hip_runtime.h>

#include <iostream>

constexpr unsigned int num_threads = 1 << 16;
constexpr unsigned int block_size = 256;
constexpr unsigned int grid_size = ceiling_div(num_threads, block_size);

__global__ void increment_std_atomic_kernel(hip::std::atomic<int> *counter,
                                            const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    counter->fetch_add(1, hip::std::memory_order_relaxed);
  }
}

__global__ void increment_device_atomic_kernel(
    hip::atomic<int, hip::thread_scope_device> *counter,
    const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    counter->fetch_add(1, hip::memory_order_relaxed);
  }
}

__global__ void increment_std_atomic_ref_kernel(int *counter,
                                                const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    hip::std::atomic_ref<int> ref(*counter);
    ref.fetch_add(1, hip::std::memory_order_relaxed);
  }
}

__global__ void increment_device_atomic_ref_kernel(int *counter,
                                                   const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    hip::atomic_ref<int, hip::thread_scope_device> ref(*counter);
    ref.fetch_add(1, hip::memory_order_relaxed);
  }
}

int main() {
  // hip::std::atomic<int> and hip::atomic<int, Scope> store nothing but a
  // single int, so hipMemset zero-initializes the counters and a plain
  // hipMemcpy into an int reads them back.
  hip::std::atomic<int> *d_std_counter{};
  HIP_CHECK(hipMalloc(&d_std_counter, sizeof(*d_std_counter)));
  HIP_CHECK(hipMemset(d_std_counter, 0, sizeof(*d_std_counter)));

  hip::atomic<int, hip::thread_scope_device> *d_device_counter{};
  HIP_CHECK(hipMalloc(&d_device_counter, sizeof(*d_device_counter)));
  HIP_CHECK(hipMemset(d_device_counter, 0, sizeof(*d_device_counter)));

  int *d_ref_counter{};
  HIP_CHECK(hipMalloc(&d_ref_counter, sizeof(*d_ref_counter)));
  HIP_CHECK(hipMemset(d_ref_counter, 0, sizeof(*d_ref_counter)));

  int *d_device_ref_counter{};
  HIP_CHECK(hipMalloc(&d_device_ref_counter, sizeof(*d_device_ref_counter)));
  HIP_CHECK(hipMemset(d_device_ref_counter, 0, sizeof(*d_device_ref_counter)));

  increment_std_atomic_kernel<<<grid_size, block_size>>>(d_std_counter,
                                                         num_threads);
  HIP_CHECK(hipGetLastError());

  increment_device_atomic_kernel<<<grid_size, block_size>>>(d_device_counter,
                                                            num_threads);
  HIP_CHECK(hipGetLastError());

  increment_std_atomic_ref_kernel<<<grid_size, block_size>>>(d_ref_counter,
                                                             num_threads);
  HIP_CHECK(hipGetLastError());

  increment_device_atomic_ref_kernel<<<grid_size, block_size>>>(
      d_device_ref_counter, num_threads);
  HIP_CHECK(hipGetLastError());

  int h_std_result{};
  int h_device_result{};
  int h_ref_result{};
  int h_device_ref_result{};
  HIP_CHECK(hipMemcpy(&h_std_result, d_std_counter, sizeof(h_std_result),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_device_result, d_device_counter,
                      sizeof(h_device_result), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_ref_result, d_ref_counter, sizeof(h_ref_result),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_device_ref_result, d_device_ref_counter,
                      sizeof(h_device_ref_result), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_std_counter));
  HIP_CHECK(hipFree(d_device_counter));
  HIP_CHECK(hipFree(d_ref_counter));
  HIP_CHECK(hipFree(d_device_ref_counter));

  std::cout << "hip::std::atomic<int>: " << h_std_result << '\n'
            << "hip::atomic<int, thread_scope_device>: " << h_device_result
            << '\n'
            << "hip::std::atomic_ref<int>: " << h_ref_result << '\n'
            << "hip::atomic_ref<int, thread_scope_device>: "
            << h_device_ref_result << std::endl;

  // Every counter must equal the number of threads that incremented it.
  int errors = 0;
  errors += (h_std_result != static_cast<int>(num_threads));
  errors += (h_device_result != static_cast<int>(num_threads));
  errors += (h_ref_result != static_cast<int>(num_threads));
  errors += (h_device_ref_result != static_cast<int>(num_threads));

  return report_validation_result(errors);
}
