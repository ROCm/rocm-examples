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

// <hip/memory_resource> provides a concept describing the minimal interface
// a memory resource must implement, a type-erased, non-owning view over any
// conforming resource, and the property tags hip::mr::device_accessible and
// hip::mr::host_accessible. Everything in the hip::mr namespace is
// experimental and therefore hidden behind the
// LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE macro, which this example
// defines on the compiler command line (see CMakeLists.txt/Makefile).
#include <hip/memory_resource>

// libhipcxx 3.4 (CCCL_VERSION >= 3004000) renamed the plain, non-stream-
// ordered resource concept/member functions/type-erased view from
// hip::mr::resource/resource_with/resource_ref/allocate/deallocate to
// hip::mr::synchronous_resource/synchronous_resource_with/
// synchronous_resource_ref/allocate_sync/deallocate_sync, freeing up the
// unprefixed names for a new stream-ordered resource concept this example
// does not use. Pick whichever spelling the installed libhipcxx provides.
#if CCCL_VERSION >= 3004000
#define HIP_MR_RESOURCE hip::mr::synchronous_resource
#define HIP_MR_RESOURCE_WITH hip::mr::synchronous_resource_with
#define HIP_MR_RESOURCE_REF hip::mr::synchronous_resource_ref
#define HIP_MR_ALLOCATE allocate_sync
#define HIP_MR_DEALLOCATE deallocate_sync
#else
#define HIP_MR_RESOURCE hip::mr::resource
#define HIP_MR_RESOURCE_WITH hip::mr::resource_with
#define HIP_MR_RESOURCE_REF hip::mr::resource_ref
#define HIP_MR_ALLOCATE allocate
#define HIP_MR_DEALLOCATE deallocate
#endif

// hip::std::has_single_bit (C++20 <bit>, provided for host and device code
// by libhipcxx) checks whether a value is a power of two; used below to
// validate the alignment requested from HipDeviceResource's allocate member.
#include <hip/std/bit>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// A minimal memory resource that allocates and frees device memory with
// hipMalloc/hipFree.
class HipDeviceResource {
public:
  // Reject an alignment hipMalloc cannot honor (non-power-of-two, or stricter
  // than the 256 bytes hip::mr::default_cuda_malloc_alignment guarantees).
  void *HIP_MR_ALLOCATE(std::size_t bytes, std::size_t alignment) {
    if (!hip::std::has_single_bit(alignment) ||
        alignment > hip::mr::default_cuda_malloc_alignment) {
      std::cerr << "HipDeviceResource: alignment " << alignment
                << " is not supported\n";
      std::exit(error_exit_code);
    }
    void *ptr{};
    HIP_CHECK(hipMalloc(&ptr, bytes));
    return ptr;
  }

  void HIP_MR_DEALLOCATE(void *ptr, std::size_t /* bytes */,
                         std::size_t /* alignment */) noexcept {
    HIP_CHECK(hipFree(ptr));
  }

  bool operator==(const HipDeviceResource &) const { return true; }
  bool operator!=(const HipDeviceResource &) const { return false; }

  // A resource declares a property by providing a friend function named
  // get_property that takes the resource and the property tag (found via ADL).
  friend constexpr void get_property(const HipDeviceResource &,
                                     hip::mr::device_accessible) noexcept {}
};

// Compile-time checks that HipDeviceResource satisfies the library concepts.
static_assert(HIP_MR_RESOURCE<HipDeviceResource>,
              "HipDeviceResource must satisfy the resource concept");
static_assert(
    HIP_MR_RESOURCE_WITH<HipDeviceResource, hip::mr::device_accessible>,
    "HipDeviceResource must declare device_accessible");

// Allocates through a type-erased resource reference, without mentioning the
// concrete resource type: this is the point of type erasure.
void *
allocate_device_buffer(HIP_MR_RESOURCE_REF<hip::mr::device_accessible> resource,
                       std::size_t bytes) {
  std::cout << "Allocating " << bytes
            << " bytes through the type-erased resource reference.\n";
  return resource.HIP_MR_ALLOCATE(bytes,
                                  hip::mr::default_cuda_malloc_alignment);
}

// Frees a buffer previously obtained from allocate_device_buffer.
void free_device_buffer(
    HIP_MR_RESOURCE_REF<hip::mr::device_accessible> resource, void *ptr,
    std::size_t bytes) {
  resource.HIP_MR_DEALLOCATE(ptr, bytes,
                             hip::mr::default_cuda_malloc_alignment);
  std::cout << "Freed " << bytes
            << " bytes through the type-erased resource reference.\n";
}

// Proves that the memory handed back by allocate_device_buffer is really
// device_accessible: it can be written from a kernel.
__global__ void fill_kernel(int *data, const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = static_cast<int>(index * index);
  }
}

int main() {
  constexpr unsigned int size = 256;
  constexpr unsigned int block_size = 256;
  constexpr unsigned int grid_size = ceiling_div(size, block_size);
  const std::size_t bytes = size * sizeof(int);

  HipDeviceResource resource;
  HIP_MR_RESOURCE_REF<hip::mr::device_accessible> ref{resource};

  void *raw_ptr = allocate_device_buffer(resource, bytes);
  int *d_data = static_cast<int *>(raw_ptr);

  fill_kernel<<<grid_size, block_size>>>(d_data, size);
  HIP_CHECK(hipGetLastError());

  std::vector<int> h_data(size);
  HIP_CHECK(hipMemcpy(h_data.data(), d_data, bytes, hipMemcpyDeviceToHost));

  free_device_buffer(ref, raw_ptr, bytes);

  // Validate that every element matches what the kernel was expected to
  // write, proving the memory was writable from device code.
  int errors = 0;
  for (unsigned int i = 0; i < size; ++i) {
    if (h_data[i] != static_cast<int>(i * i)) {
      ++errors;
    }
  }

  return report_validation_result(errors);
}
