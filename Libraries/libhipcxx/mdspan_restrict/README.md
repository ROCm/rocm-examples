# libhipcxx Mdspan Restrict Example

## Description

This example shows how `cuda::std::mdspan` lets a kernel take a multidimensional view of data instead of a raw pointer plus separate size parameters, and how `cuda::restrict_mdspan` carries the no-aliasing promise of a `__restrict__` pointer in that view's *type*.

A single kernel template, `add_matrices_kernel`, computes `c(i, j) = a(i, j) + b(i, j)` for an 8x12 matrix. It is instantiated twice from the same source:

- Once with plain `cuda::std::mdspan` parameters.
- Once with `cuda::restrict_mdspan` parameters, which wrap the same accessor type in `cuda::restrict_accessor` so that the underlying data handle becomes `element_type * __restrict`.

The example also builds two `cuda::std::mdspan` views with different `LayoutPolicy` template arguments (`cuda::std::layout_right`, the default, and `cuda::std::layout_left`) over the *same* host buffer, to show that the layout policy -- not just the extents -- determines which buffer element a given `(i, j)` index refers to.

### Application flow

1. Two `8 x 12` host input matrices are filled with distinct, per-element values so that indexing mistakes would be visible in the printed output.
2. Device memory is allocated for the two inputs and for two separate output buffers (one per kernel instantiation), and the inputs are copied to the device.
3. `add_matrices_kernel` is instantiated with plain `cuda::std::mdspan<const float, Extents>` / `cuda::std::mdspan<float, Extents>` parameters and launched with a 2D grid of `4x4` thread blocks; this is a plain element-wise add, and each `4x4` thread block simply covers one `4x4` tile of the output matrix, not a tiled (shared-memory-staged) algorithm. Every `mdspan` is built with the `mdspan(data_handle_type p, const extents_type& ext)` constructor.
4. The same `add_matrices_kernel` template is instantiated again, this time with `cuda::restrict_mdspan<const float, Extents>` / `cuda::restrict_mdspan<float, Extents>` parameters, and launched the same way.
5. Both results are copied back to the host, a small top-left slice of the plain-mdspan result is printed, and both results are compared against a host-computed reference.
6. Two `cuda::std::mdspan` views (one `layout_right`, one `layout_left`) are built over the same host input buffer. For a few `(i, j)` indices, the example prints and checks the linear offset each layout computes via `mapping()(i, j)`, showing that the two layouts address different elements of the same buffer.
7. The number of mismatches from steps 5 and 6 is printed to the standard output.

## Key APIs and Concepts

- `cuda::std::mdspan<T, Extents, Layout = layout_right, Accessor = default_accessor<T>>` is a non-owning view that combines a data handle with a multidimensional shape (`Extents`) and an indexing scheme (`Layout`), so a kernel can take one `mdspan` parameter instead of a pointer and one size parameter per dimension.
- `cuda::std::extents<IndexType, Exts...>` describes that shape. Every extent in this example is given as a compile-time size (`cuda::std::extents<size_t, 8, 12>`), so `Extents::rank_dynamic() == 0` and an `Extents` value can be default-constructed, including with the `{}` shorthand: `mdspan(ptr, {})` list-initializes the `const extents_type&` constructor argument from `Extents`'s default constructor.
- `mdspan(data_handle_type p, const extents_type& ext)` is the constructor used throughout this example to build an `mdspan` (or `restrict_mdspan`) from a device or host pointer and an `Extents` value.
- `md(i, j)` calls `mdspan::operator()`, a libhipcxx/Kokkos-`mdspan` extension that accepts one index per dimension directly (in the C++23 Standard this indexing is spelled `md[i, j]`); it is bounds-checked with `_CCCL_ASSERT` in debug builds.
- `cuda::restrict_mdspan<T, Extents, Layout = layout_right, Accessor = default_accessor<T>>` is a libhipcxx alias for `cuda::std::mdspan<T, Extents, Layout, cuda::restrict_accessor<Accessor>>`: the same `mdspan`, with its accessor swapped out.
- `cuda::restrict_accessor<Accessor>` wraps an accessor whose `data_handle_type` is already a pointer and redeclares `data_handle_type` as `element_type * __restrict`. That is where the no-aliasing promise lives: it is part of the kernel parameter's *type*, so it survives the move from `(T *__restrict__, size_t rows, size_t cols)`-style parameters to `mdspan` parameters without needing a `__restrict__` keyword anywhere in the kernel signature.
- `cuda::std::layout_right` (the default `LayoutPolicy`) is row-major: the *last* index varies fastest, so `md(i, j)` maps to linear offset `i * cols + j`. `cuda::std::layout_left` is column-major: the *first* index varies fastest, mapping `md(i, j)` to offset `i + j * rows`. Both are drop-in `LayoutPolicy` template arguments; swapping one for the other over the same buffer changes which element a given `(i, j)` refers to, as this example's `mapping()(i, j)` comparison shows.

## Demonstrated API Calls

### libhipcxx

- `cuda::std::mdspan`
- `cuda::std::extents`
- `cuda::std::layout_right`
- `cuda::std::layout_left`
- `cuda::restrict_mdspan`
- `cuda::restrict_accessor` (indirectly, as the accessor of `cuda::restrict_mdspan`)

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
