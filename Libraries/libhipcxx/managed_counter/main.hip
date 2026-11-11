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

// <hip/atomic> provides hip::std::atomic_ref, which layers atomic access
// over a plain, non-atomic int. The int below lives in managed memory, so
// host threads and device threads can increment the same counter at the
// same time.
#include <hip/atomic>

#include <hip/hip_runtime.h>

#include <iostream>
#include <new>
#include <thread>
#include <vector>

constexpr unsigned int host_threads = 2;
constexpr unsigned int host_increments = 10000; // per host thread
constexpr unsigned int device_increments = 10000;
constexpr unsigned int expected_total =
    host_threads * host_increments + device_increments;
constexpr unsigned int block_size = 256;
constexpr unsigned int grid_size = ceiling_div(device_increments, block_size);

// Each host thread increments *counter num_increments times.
void host_increment_task(int *counter, const unsigned int num_increments) {
  hip::std::atomic_ref<int> ref(*counter);
  for (unsigned int i = 0; i < num_increments; ++i) {
    ref.fetch_add(1, hip::std::memory_order_relaxed);
  }
}

// Every device thread increments *counter once.
__global__ void increment_kernel(int *counter) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < device_increments) {
    hip::std::atomic_ref<int> ref(*counter);
    ref.fetch_add(1, hip::std::memory_order_relaxed);
  }
}

int main() {
  // Guard against devices that do not report managed-memory support.
  int device = 0;
  int managed_memory_supported = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&managed_memory_supported,
                                  hipDeviceAttributeManagedMemory, device));
  if (!managed_memory_supported) {
    std::cout << "Skipping, device " << device
              << " does not support managed memory" << std::endl;
    return 0;
  }

  // Allocate one managed page: the host and the device share it, so no
  // hipMemcpy is needed. hipMallocManaged does not run a constructor, so
  // the counter is zero-initialized in place with placement new.
  int *counter{};
  HIP_CHECK(hipMallocManaged(&counter, sizeof(int)));
  ::new (counter) int{0};

  // Launch the device increments and interleave them with the host
  // increments, so that both sides really run in parallel.
  std::vector<std::thread> threads;
  threads.emplace_back(host_increment_task, counter, host_increments);
  increment_kernel<<<grid_size, block_size>>>(counter);
  HIP_CHECK(hipGetLastError());
  threads.emplace_back(host_increment_task, counter, host_increments);

  for (auto &thread : threads) {
    thread.join();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::cout << host_threads << " host threads x " << host_increments
            << " increments + " << device_increments
            << " device increments = " << *counter << '\n';

  // Every increment must have been counted, no matter which side did it.
  int errors = (*counter != static_cast<int>(expected_total));

  HIP_CHECK(hipFree(counter));

  return report_validation_result(errors);
}
