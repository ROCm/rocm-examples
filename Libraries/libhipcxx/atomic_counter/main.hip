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

// Every thread below increments a counter exactly once. Because the
// increments happen concurrently, a plain (non-atomic) int would lose
// updates to a data race; each of the four counters below uses a different
// libhipcxx atomic spelling to make the increment safe, and afterwards must
// equal exactly this value.
constexpr unsigned int num_threads = 1 << 16;

// Threads per block and number of blocks, shared by every kernel below.
constexpr unsigned int block_size = 256;
constexpr unsigned int grid_size = ceiling_div(num_threads, block_size);

/// \brief Every thread increments *counter by one.
///
/// hip::std::atomic<int> is the conforming C++ Standard atomic: the spelling
/// is identical to host-only std::atomic<int>, only the include changes
/// (<hip/atomic> instead of <atomic>), and the same object can be
/// incremented from __host__ and __device__ code alike.
///
/// Every fetch_add below passes hip::std::memory_order_relaxed (or, for the
/// hip::atomic kernels further down, the equivalent
/// hip::memory_order_relaxed): a pure counter needs atomicity - no increment
/// may be lost - but not ordering with any other memory operation, so relaxed
/// is the idiomatic, cheapest-available choice. The final value is only ever
/// read after the kernel has completed, and hipMemcpy synchronizes with the
/// device, so no stronger order is needed to observe it correctly on the host.
__global__ void increment_std_atomic_kernel(hip::std::atomic<int> *counter,
                                            const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    counter->fetch_add(1, hip::std::memory_order_relaxed);
  }
}

/// \brief The same counter, expressed with hip::atomic instead of
/// hip::std::atomic.
///
/// hip::atomic<T, Scope> is a libhipcxx extension of hip::std::atomic that
/// adds a second, optional template parameter: a hip::thread_scope, i.e. the
/// set of threads whose accesses to the atomic are guaranteed to synchronize
/// with each other. It defaults to hip::thread_scope_system - the same scope
/// hip::std::atomic is implicitly locked to - so hip::atomic<int> without a
/// second argument and hip::std::atomic<int> are the same type in every way
/// that matters here: same layout, same operations, same behavior. The
/// extension is the *ability to narrow* the scope, not a different default.
/// This kernel narrows to hip::thread_scope_device because every increment
/// below happens from device threads only; a narrower scope lets the
/// compiler avoid synchronization that would only be needed to also
/// coordinate with the host.
///
/// This kernel and the two below use the hip:: (not hip::std::) spelling
/// family, so they use hip::memory_order_relaxed for consistency, even
/// though it names the same enumerator as hip::std::memory_order_relaxed.
__global__ void increment_device_atomic_kernel(
    hip::atomic<int, hip::thread_scope_device> *counter,
    const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    counter->fetch_add(1, hip::memory_order_relaxed);
  }
}

/// \brief Increments a *plain*, non-atomic int through a
/// hip::std::atomic_ref<int>.
///
/// Unlike hip::std::atomic<int>, an atomic_ref does not own the storage it
/// protects: *counter is an ordinary int, allocated and zero-initialized like
/// any other value, with no atomic object ever constructed over it.
/// atomic_ref only makes the accesses that go *through* the reference atomic;
/// any other access to the same int that does not go through a live
/// atomic_ref referring to it would still be a data race.
__global__ void increment_std_atomic_ref_kernel(int *counter,
                                                const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    hip::std::atomic_ref<int> ref(*counter);
    ref.fetch_add(1, hip::std::memory_order_relaxed);
  }
}

/// \brief The same atomic_ref pattern, narrowed to hip::thread_scope_device
/// with hip::atomic_ref<int, Scope> - the scoped counterpart of
/// hip::std::atomic_ref, for the same reason as
/// increment_device_atomic_kernel above.
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
  // single int: cuda/std/__atomic/types/base.h defines __atomic_storage<Tp>
  // with exactly one non-static data member, _Tp __a_value, and int is large
  // enough (4 bytes) that libhipcxx never routes it through the padded
  // "small type" or lock-based storage instead. hipMalloc'd memory has had no
  // constructor run over it, but zero is a valid bit pattern for a
  // freshly-constructed int, so hipMemset to all-zero bytes is a valid way to
  // initialize each counter in place, without an extra kernel or a placement
  // new.
  hip::std::atomic<int> *d_std_counter{};
  HIP_CHECK(hipMalloc(&d_std_counter, sizeof(*d_std_counter)));
  HIP_CHECK(hipMemset(d_std_counter, 0, sizeof(*d_std_counter)));

  hip::atomic<int, hip::thread_scope_device> *d_device_counter{};
  HIP_CHECK(hipMalloc(&d_device_counter, sizeof(*d_device_counter)));
  HIP_CHECK(hipMemset(d_device_counter, 0, sizeof(*d_device_counter)));

  // The atomic_ref kernels operate on plain ints, which need the same
  // zero-initialization.
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

  // Every counter is, in memory, nothing but a single int (see above), so a
  // bit-for-bit copy into a plain host int is all that is needed to read back
  // the result; no atomic object is created on the host.
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

  // Print the four results so that they can be checked by eye: every thread
  // performed exactly one increment, so every counter must equal
  // num_threads.
  std::cout << "Threads per counter: " << num_threads << "\n\n";
  std::cout << "hip::std::atomic<int>                                     -> "
            << h_std_result << '\n';
  std::cout << "hip::atomic<int, thread_scope_device>                     -> "
            << h_device_result << '\n';
  std::cout << "hip::std::atomic_ref<int> over a plain int                -> "
            << h_ref_result << '\n';
  std::cout << "hip::atomic_ref<int, thread_scope_device> over a plain int -> "
            << h_device_ref_result << '\n';
  std::cout << std::endl;

  // Validate that every counter reached exactly num_threads, i.e. that no
  // increment was lost to a data race.
  int errors = 0;
  errors += (h_std_result != static_cast<int>(num_threads));
  errors += (h_device_result != static_cast<int>(num_threads));
  errors += (h_ref_result != static_cast<int>(num_threads));
  errors += (h_device_ref_result != static_cast<int>(num_threads));

  return report_validation_result(errors);
}
