# libhipcxx Memory Resource Example

## Description

This example shows how to implement a custom memory resource that satisfies libhipcxx's plain (non-stream-ordered) resource concept, and how to consume it through the corresponding type-erased, non-owning resource reference. `hip::mr` (the resource concept, the type-erased resource reference, the `device_accessible`/`host_accessible` property tags, and `get_property`) is an experimental part of libhipcxx and is only compiled in when `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` is defined; both the CMake and Makefile builds of this example define it for you.

Starting with libhipcxx 3.4 (`CCCL_VERSION >= 3004000`), the plain, non-stream-ordered resource concept, its member functions, and its type-erased view have new names:

| Before 3.4 | From 3.4 on |
| --- | --- |
| `hip::mr::resource` | `hip::mr::synchronous_resource` |
| `hip::mr::resource_with` | `hip::mr::synchronous_resource_with` |
| `hip::mr::resource_ref` | `hip::mr::synchronous_resource_ref` |
| `allocate`, `deallocate` | `allocate_sync`, `deallocate_sync` |

The rename frees the unprefixed names for a new stream-ordered resource concept, which this example does not use. `main.hip` selects whichever names the installed libhipcxx provides through a small `HIP_MR_*` macro shim keyed on `CCCL_VERSION`, so the example builds against libhipcxx both before 3.4 and from 3.4 on.

The example defines `HipDeviceResource`, a minimal resource whose allocate/deallocate member functions call `hipMalloc`/`hipFree`, and which declares itself `device_accessible` through a friend `get_property` function. It is then wrapped in a type-erased resource reference over `hip::mr::device_accessible` and passed to plain (non-templated) functions that allocate and free a device buffer without ever naming `HipDeviceResource`. A kernel writes into the allocated buffer to demonstrate that the memory really is device-accessible, and the result is copied back and validated.

### Application flow

1. `HipDeviceResource` is defined. Its allocate member checks that the requested alignment is a power of two and does not exceed `hip::mr::default_cuda_malloc_alignment` (256 bytes, the alignment `hipMalloc` guarantees), reporting an error and exiting if not, then forwards to `hipMalloc`; its deallocate member forwards to `hipFree`. The type is equality-comparable (it has no state, so all instances compare equal), and it declares the `device_accessible` property via a friend `get_property` function.
2. Two `static_assert`s check, at compile time, that `HipDeviceResource` conforms to the resource concept and declares `device_accessible`.
3. An instance of `HipDeviceResource` is created and wrapped in a type-erased resource reference over `hip::mr::device_accessible`.
4. `allocate_device_buffer`, a function that only takes that type-erased resource reference (not a template), allocates a device buffer through it.
5. The `fill_kernel` kernel is launched with one thread per element and writes a value derived from each thread's global index into the buffer.
6. The results are copied back to the host, and `free_device_buffer` frees the buffer, again only through the type-erased resource reference.
7. Every element is validated against the value the kernel was expected to write, and the result of the comparison is printed to the standard output.

## Key APIs and Concepts

- The plain resource concept (`hip::mr::resource` before libhipcxx 3.4, `hip::mr::synchronous_resource` from 3.4 on) is satisfied by a type that is equality-comparable (`operator==` and `operator!=`) and provides these member functions:
  - `void* allocate(size_t bytes, size_t alignment)` (`allocate_sync` from 3.4 on)
  - `void deallocate(void* ptr, size_t bytes, size_t alignment)` (`deallocate_sync` from 3.4 on)

  The corresponding `hip::mr::resource_with<Resource, Properties...>` concept (`hip::mr::synchronous_resource_with` from 3.4 on) additionally requires that `Resource` has every property in `Properties...`.
- A resource declares a property by providing a friend function named `get_property` that takes the resource and the property tag. `hip::mr::device_accessible` and `hip::mr::host_accessible` are stateless tags (empty structs), so their `get_property` overload returns `void`; the overload's existence, found via argument-dependent lookup, is what signals the property.
  - `device_accessible` tells a caller that memory returned by the resource's allocate member can be read and written from device code (kernels).
  - `host_accessible` tells a caller that the memory can be read and written from host code. A resource can declare either, both, or (as in this example) only one of them.
- The corresponding type-erased resource reference (`hip::mr::resource_ref<Properties...>` before libhipcxx 3.4, `hip::mr::synchronous_resource_ref<Properties...>` from 3.4 on) is a non-owning, type-erased view over any type that conforms to the plain resource concept and has (at least) the given properties. It is constructible from a `Resource&` or `Resource*`, and exposes allocate/deallocate member functions (with `alignment` defaulting to `alignof(std::max_align_t)` if omitted), forwarding every call to the wrapped resource through an internal vtable.
- `hip::mr::default_cuda_malloc_alignment` (`cuda/__memory_resource/properties.h`) is the alignment (256 bytes) that `hipMalloc` guarantees for its returned memory. `HipDeviceResource`'s allocate member uses it, together with `hip::std::has_single_bit`, to reject an alignment request it cannot satisfy instead of silently under-aligning the memory it returns.
- Type erasure via the resource reference means a function that only needs to allocate/deallocate memory can take a resource reference parameter instead of being templated on the concrete resource type. Such a function accepts `HipDeviceResource`, a pool allocator, a logging wrapper, or any other conforming resource, all without recompiling or even needing to know those types exist.

## Demonstrated API Calls

### libhipcxx

- `hip::mr::resource`/`hip::mr::synchronous_resource`
- `hip::mr::resource_with`/`hip::mr::synchronous_resource_with`
- `hip::mr::resource_ref`/`hip::mr::synchronous_resource_ref`
- `hip::mr::device_accessible`
- `hip::mr::default_cuda_malloc_alignment`
- `hip::std::has_single_bit`
- `get_property` (as a friend function of a custom resource)

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
