# libhipcxx Index Math Example

## Description

This example shows how to use the libhipcxx index-math helpers `hip::ceil_div`, `hip::round_up`, and `hip::round_down` from `<hip/cmath>` for two everyday GPU tasks: sizing a kernel launch grid, and aligning buffer sizes and offsets to a tile boundary. All three helpers are `constexpr`, callable from both host and device code, and are libhipcxx extensions, not part of the C++ Standard.

- `hip::ceil_div(a, b)` divides `a` by `b`, rounding up if there is a remainder. It is used here to compute the number of blocks needed to cover a problem size that is not an exact multiple of the block size.
- `hip::round_up(a, b)` rounds `a` up to the next multiple of `b`. It is used here to round a buffer size up to a tile boundary before allocating.
- `hip::round_down(a, b)` rounds `a` down to the previous multiple of `b`. It is used here to find the start of the tile that contains a given offset.

### Application flow

1. `hip::ceil_div` computes the grid size for a problem size that is not a multiple of the block size, so the launch grid overshoots the problem size.
2. A trivial kernel is launched with that grid size. Each thread writes its global index to device memory, guarded by a bounds check against the problem size so that the overshoot threads do nothing.
3. The results are copied back to the host, freed on the device, and checked against the expected indices.
4. A compile-time and a runtime comparison show why the naive `(a + b - 1) / b` idiom for computing a ceiling division is not a safe replacement for `hip::ceil_div`: for a dividend near the maximum value of its type, the naive idiom overflows and silently produces a wrong result, while `hip::ceil_div` does not.
5. `hip::round_up` rounds a buffer size up to a tile boundary before allocating, and `hip::round_down` rounds an offset down to the start of its containing tile.
6. A table of boundary cases (zero, one below a tile boundary, exactly on a boundary, and a couple of larger values) is printed for both `hip::round_up` and `hip::round_down`.
7. Every value in the table is checked against an independent host oracle that uses plain division and a remainder check, not the naive `(a + b - 1) / b` idiom, and the result of the comparison is printed to the standard output.

## Key APIs and Concepts

- `hip::ceil_div(a, b)` requires `b` to be positive, and, if `a`'s type is signed, `a` to be non-negative. It returns the common type of its two arguments. `include/cuda/__cmath/ceil_div.h` selects a different implementation depending on where the code runs, but only when `a / b`'s promoted type is unsigned (the common case for sizes): it then uses `NV_IF_ELSE_TARGET` to choose between a `min`-based formula on the device (a single division, no remainder) and a division-with-remainder-check formula on the host. The header notes that the `min`-based method is faster even when the divisor is a compile-time constant. When `a / b`'s promoted type is signed instead, host and device both use the same `(a + b - 1) / b`-style formula, computed after casting to the corresponding unsigned type to sidestep overflow -- there is no separate device path in that case. Both formulas avoid the overflow that a hand-rolled `(a + b - 1) / b` idiom over the *original* (non-promoted) type is prone to for a dividend near the maximum value of its type.
- `hip::round_up(a, b)` and `hip::round_down(a, b)` have the same preconditions as `hip::ceil_div` (`b > 0`, and `a >= 0` if `a`'s type is signed) and also return the common type of their arguments. `hip::round_up` rounds `a` up to the next multiple of `b`; an already-aligned value is left unchanged. `hip::round_down` rounds `a` down to the previous multiple of `b`; `hip::round_down(0, b)` is `0`.
- All three functions accept mixed integer types, mixed signedness, and enum operands with an underlying integer type (not used in this example, since sizes here are simple `unsigned int` values); the return type is deduced from the operands via `common_type`.

## Demonstrated API Calls

### libhipcxx

- `hip::ceil_div`
- `hip::round_up`
- `hip::round_down`

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
