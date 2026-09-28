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

// <cuda/memory_resource> provides cuda::mr::resource (a concept describing
// the minimal interface a memory resource must implement),
// cuda::mr::resource_ref (a type-erased, non-owning view over any conforming
// resource), and the property tags cuda::mr::device_accessible and
// cuda::mr::host_accessible. Everything in the cuda::mr namespace is
// experimental and therefore hidden behind the
// LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE macro, which this example
// defines on the compiler command line (see CMakeLists.txt/Makefile).
#include <cuda/memory_resource>

// cuda::std::has_single_bit (C++20 <bit>, provided for host and device code
// by libhipcxx) checks whether a value is a power of two; used below to
// validate the alignment requested from HipDeviceResource::allocate.
#include <cuda/std/bit>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

/// \brief A minimal memory resource that allocates and frees device memory
/// with hipMalloc/hipFree.
///
/// To satisfy the cuda::mr::resource concept
/// (cuda/__memory_resource/resource.h) a type must provide:
///   - `void* allocate(size_t bytes, size_t alignment)`
///   - `void deallocate(void* ptr, size_t bytes, size_t alignment)`
///   - `operator==` and, for compatibility with pre-C++20 rules,
///     `operator!=` (the concept requires equality_comparable, which does not
///     synthesize operator!= from operator== the way C++20 does).
///
/// This resource has no state, so every instance compares equal to every
/// other instance. hipMalloc always aligns its returned memory to
/// cuda::mr::default_cuda_malloc_alignment (256 bytes,
/// cuda/__memory_resource/properties.h), so allocate() below can honor any
/// requested alignment up to that value, but not a stricter one.
class HipDeviceResource {
public:
  void *allocate(size_t bytes, size_t alignment) {
    // Reject a request that hipMalloc cannot honor, rather than silently
    // handing back memory that is not aligned as strictly as the caller
    // asked for: a non-power-of-two alignment is meaningless, and a
    // power-of-two alignment stricter than 256 bytes exceeds what hipMalloc
    // guarantees.
    if (!cuda::std::has_single_bit(alignment) ||
        alignment > cuda::mr::default_cuda_malloc_alignment) {
      std::cerr << "HipDeviceResource::allocate: alignment " << alignment
                << " is not supported (must be a power of two and at most "
                << cuda::mr::default_cuda_malloc_alignment << ")\n";
      std::exit(error_exit_code);
    }
    void *ptr{};
    HIP_CHECK(hipMalloc(&ptr, bytes));
    return ptr;
  }

  void deallocate(void *ptr, size_t /* bytes */,
                  size_t /* alignment */) noexcept {
    HIP_CHECK(hipFree(ptr));
  }

  bool operator==(const HipDeviceResource &) const { return true; }
  bool operator!=(const HipDeviceResource &) const { return false; }

  // A resource declares a property by providing a friend function named
  // get_property that takes the resource and the property tag. cuda::mr::
  // device_accessible and cuda::mr::host_accessible are stateless tags (empty
  // structs), so the matching get_property overload simply returns void; its
  // existence is the signal, not its return value. get_property is found via
  // argument-dependent lookup, which is why it must be declared as a friend
  // rather than a private member function.
  //
  // device_accessible tells a caller that memory returned by allocate() can
  // be read and written from device code (kernels). It says nothing about
  // host accessibility: HipDeviceResource intentionally does not declare
  // host_accessible, because the pointers it returns must not be
  // dereferenced from host code.
  friend constexpr void get_property(const HipDeviceResource &,
                                     cuda::mr::device_accessible) noexcept {}
};

// HipDeviceResource is checked against the library concepts at compile
// time. cuda::mr::resource_with additionally verifies that the given
// properties are satisfied via get_property/has_property.
static_assert(cuda::mr::resource<HipDeviceResource>,
              "HipDeviceResource must satisfy the cuda::mr::resource "
              "concept: allocate/deallocate with the right signatures, plus "
              "operator== and operator!=");
static_assert(
    cuda::mr::resource_with<HipDeviceResource, cuda::mr::device_accessible>,
    "HipDeviceResource must additionally declare device_accessible");

/// \brief Allocates `bytes` bytes through a type-erased resource_ref.
///
/// This function is not templated on the concrete resource type and never
/// mentions HipDeviceResource: `resource` could just as well be backed by
/// a pool allocator, a logging wrapper, or any other type that conforms to
/// cuda::mr::resource and declares device_accessible. This is the point of
/// type erasure via resource_ref -- code that only needs to allocate memory
/// does not need to become a template, and does not need to know (or even
/// forward-declare) every resource type it might be handed.
///
/// The alignment argument is omitted here, so resource_ref::allocate uses its
/// default of alignof(std::max_align_t).
void *allocate_device_buffer(
    cuda::mr::resource_ref<cuda::mr::device_accessible> resource,
    size_t bytes) {
  std::cout << "Allocating " << bytes
            << " bytes through cuda::mr::resource_ref<device_accessible>...\n";
  return resource.allocate(bytes);
}

/// \brief Frees a buffer previously obtained from allocate_device_buffer,
/// again purely through the type-erased view.
void free_device_buffer(
    cuda::mr::resource_ref<cuda::mr::device_accessible> resource, void *ptr,
    size_t bytes) {
  resource.deallocate(ptr, bytes);
  std::cout << "Freed " << bytes << " bytes through cuda::mr::resource_ref.\n";
}

/// \brief Each thread writes a value derived from its global index into the
/// buffer, proving that the memory handed back by resource_ref::allocate is
/// really device_accessible: it can be written from a kernel.
__global__ void fill_kernel(int *data, const unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = static_cast<int>(index * index);
  }
}

int main() {
  // Number of ints in the buffer.
  constexpr unsigned int size = 256;

  // Number of threads per block.
  constexpr unsigned int block_size = 256;

  // Number of blocks, rounded up so that every element gets a thread.
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  const size_t bytes = size * sizeof(int);

  // The concrete resource. Its type is only known here, at the point where it
  // is created -- everything downstream only sees it through resource_ref.
  HipDeviceResource resource;

  // Wrap the resource in a type-erased resource_ref<device_accessible>. This
  // is "the consumer's view": ref knows how to allocate/deallocate through
  // whatever resource it wraps, and that the resulting memory is
  // device-accessible, but nothing else about the resource's type.
  //
  // resource_ref is a non-owning view (a "fat pointer" plus a small vtable):
  // it does not manage the lifetime of `resource`, which must outlive it.
  cuda::mr::resource_ref<cuda::mr::device_accessible> ref{resource};

  // allocate_device_buffer/free_device_buffer take a resource_ref by value.
  // Passing `resource` here relies on resource_ref's non-explicit,
  // converting constructor from `Resource&`; passing `ref` itself works
  // exactly the same way, since resource_ref converts to another
  // resource_ref with the same (or a compatible) set of properties.
  void *raw_ptr = allocate_device_buffer(resource, bytes);
  int *d_data = static_cast<int *>(raw_ptr);

  // Launch a trivial kernel that writes into the allocated buffer.
  fill_kernel<<<grid_size, block_size>>>(d_data, size);

  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

  // Copy the results back to the host so they can be printed and validated.
  std::vector<int> h_data(size);
  HIP_CHECK(hipMemcpy(h_data.data(), d_data, bytes, hipMemcpyDeviceToHost));

  // Print the first few values so that the result can be checked by eye.
  constexpr unsigned int print_count = 8;
  std::cout << "First " << print_count << " values written by the device:\n";
  for (unsigned int i = 0; i < print_count; ++i) {
    std::cout << "  h_data[" << i << "] = " << h_data[i] << '\n';
  }

  // Free the buffer, again only through the type-erased ref.
  free_device_buffer(ref, raw_ptr, bytes);

  // Validate that every element matches what the kernel was expected to
  // write. This also confirms that the memory returned by resource_ref,
  // backed by HipDeviceResource's hipMalloc call, really was writable from
  // device code.
  int errors = 0;
  for (unsigned int i = 0; i < size; ++i) {
    if (h_data[i] != static_cast<int>(i * i)) {
      ++errors;
    }
  }

  return report_validation_result(errors);
}
