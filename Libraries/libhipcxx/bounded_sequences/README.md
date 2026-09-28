# libhipcxx Bounded Sequences Example

## Description

This example shows three "bounded sequence" types from libhipcxx: containers and views whose storage requirements are fixed and known up front, so they need no dynamic memory allocation and work equally well in registers, in global memory, and in shared memory (LDS):

- `cuda::std::span`: a non-owning view over a contiguous range, bundling a pointer and a length into a single object.
- `cuda::std::inplace_vector`: a fixed-capacity, allocator-free sequence container. It is a backport of a C++26 Standard Library addition, so most readers will not have seen it before; it behaves like `std::vector`, except that its capacity is a template parameter and reaching that capacity does not reallocate.
- `cuda::std::array`: a fixed-size array with `std::tuple`-like access through `get<N>` and structured bindings.

A kernel takes the whole input as a `cuda::std::span<const int>` instead of a `(pointer, length)` pair. Each block slices its own tile out of that span with `subspan`, scans it, and collects the values that satisfy a predicate ("is even") into a `cuda::std::inplace_vector` that lives in shared memory. The container's capacity (`max_hits_per_block`, 8) is deliberately smaller than the tile size (`tile_size`, 16), so a tile can contain more hits than the container can hold -- which is the interesting case this example is built to show.

### Application flow

1. The host builds an array of 100 `int`s from an index-derived formula. Two tiles are overwritten on purpose: block 0's tile is made entirely even (so every element in it is a hit, more than fit in the capacity-8 container), and block 2's tile is made entirely odd (so it has zero hits).
2. The input is copied to the device and wrapped in a `cuda::std::span<const int>`.
3. `find_hits_kernel` is launched with one block per tile of the input (the last tile is partial, since 100 is not a multiple of the tile size). Each block:
   1. Slices its tile out of the full span with `subspan`, clamping the tile length for the last, partial tile.
   2. Reserves raw, correctly aligned `__shared__` storage for one `inplace_vector`, and has thread 0 alone construct the container into that storage with placement new (see "Key APIs and Concepts" for why a plain `__shared__ inplace_vector` variable does not compile).
   3. Has thread 0 scan the tile serially and call `try_emplace_back` for every even value. `try_emplace_back` returns a pointer to the new element, or a null pointer if the container was already at capacity. Both the number of stored hits and the number of failed (overflow) attempts are counted.
   4. Has thread 0 write the block's hit count, overflow count, and stored values back to device memory, then destroy the container again.
4. The results are copied back to the host and the device memory is freed.
5. A table with one row per block is printed: the tile's index range, its hit count, its overflow count, and the stored hit values.
6. The host recomputes, independently and without using `cuda::std::inplace_vector`, which values of each tile are hits and how many would overflow a capacity-8 container, and compares that to the device results.
7. A short, separate demonstration builds a `cuda::std::array<unsigned int, 3>` from the totals, reads it back with `cuda::std::get<N>`, and again with a structured binding, and checks that both ways of reading it agree.

## Key APIs and Concepts

- `cuda::std::span<T>` is a view, not a container: it stores a pointer and a size, and does not own or copy the underlying data. Passing one to a kernel is passing that pointer-and-size pair as a single, self-describing argument.
- `span::subspan(offset, count)` returns a view over `[offset, offset + count)` of the original span. Unlike raw pointer arithmetic, it is exception/assertion-guarded and the resulting span still knows its own size, so a per-block slice is always used with its own correct bounds.
- `cuda::std::inplace_vector<T, Capacity>` stores its elements directly inside the object -- there is no pointer to separately allocated storage, and no allocator anywhere. Its size can vary at run time between 0 and `Capacity`, but it can never exceed `Capacity`. This is what makes it usable in `__shared__` memory: shared memory is a fixed-size, per-block resource, and a container that could try to grow past a compile-time bound simply has no way to allocate more of it.
- Growing an `inplace_vector` past its capacity is a checked condition, not undefined behavior, but the exact reaction depends on which member function is used:
  - `push_back`/`emplace_back` throw `std::bad_alloc` on the host. On the device, where C++ exceptions are not available, libhipcxx's `__throw_bad_alloc` calls `cuda::std::terminate()` instead of throwing. Calling `emplace_back` on a full `inplace_vector` in device code is therefore fatal to the kernel, not silently incorrect -- but it is not something this example wants to rely on.
  - `try_emplace_back`/`try_push_back` never throw or terminate: they return a pointer to the newly constructed element on success, or a null pointer if the container was already at capacity, leaving it unchanged. This is the interface used here, precisely because the example deliberately drives some tiles to (and past) the container's capacity and needs a defined, checkable outcome for that case rather than a kernel abort.
  - `unchecked_emplace_back` skips the capacity check entirely and is undefined behavior if the container is already full; this example never calls it directly.
- `__shared__` variables in HIP cannot have a non-empty constructor: a plain `__shared__ cuda::std::inplace_vector<int, max_hits_per_block> hits;` does not compile, because `inplace_vector<int, N>`'s default constructor is not empty -- it zero-initializes the element storage and the size counter. The fix used here is to declare `__shared__` storage as raw, correctly aligned bytes (`alignas(HitVector) __shared__ unsigned char hits_storage[sizeof(HitVector)];`) and have a single thread construct the `inplace_vector` into it explicitly with placement new, then destroy it explicitly once done. This also underlines the point of `inplace_vector`: it needs only raw storage of the right size and alignment to come alive, not an allocator.
- `cuda::std::array<T, N>` supports the same `tuple_size`/`tuple_element`/`get<N>` protocol as `std::array`, so both `cuda::std::get<N>(a)` and structured bindings (`auto [x, y, z] = a;`) work to read its elements by position.

## Demonstrated API Calls

### libhipcxx

- `cuda::std::span<T>`
- `cuda::std::span<T>::subspan`
- `cuda::std::span<T>::size`
- `cuda::std::inplace_vector<T, Capacity>`
- `cuda::std::inplace_vector<T, Capacity>::try_emplace_back`
- `cuda::std::inplace_vector<T, Capacity>::size`
- `cuda::std::inplace_vector<T, Capacity>::operator[]`
- `cuda::std::array<T, N>`
- `cuda::std::get<N>`

### HIP runtime

#### Device symbols

- `blockIdx`
- `threadIdx`
- `__shared__`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
