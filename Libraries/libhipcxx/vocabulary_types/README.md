# libhipcxx Vocabulary Types Example

## Description

This example shows how the "vocabulary types" of libhipcxx -- `cuda::std::tuple`, `cuda::std::pair`, `cuda::std::expected`, and `cuda::std::optional` -- are used together to cross the host/device boundary. A tuple built once on the host is passed unchanged into a kernel launch; each thread reads it, combines it with its own thread index, and returns a `cuda::std::expected` that is either a successful floating-point result or one of two error codes. An optional intermediate value is computed by some threads and not others. Finally, HIP's built-in `float4` vector type is used as a tuple through `cuda::std::get<N>`, and a `cuda::std::pair` is round-tripped through the host's `std::pair`.

### Application flow

1. The tuple interface of `float4` is checked at compile time with `static_assert`: `cuda::std::tuple_size<float4>` is 4, and `cuda::std::tuple_element<0, float4>::type` is `float`.
2. A configuration tuple `cuda::std::tuple<int, float, float4>` -- a numerator, a scale factor, and a `float4` of per-channel weights -- is built once on the host with `cuda::std::make_tuple`.
3. Device memory is allocated for one result per thread.
4. The `analyze_records_kernel` kernel is launched with one thread per record, taking the configuration tuple as a by-value kernel argument. Each thread:
   1. Unpacks the configuration with `cuda::std::get<0/1/2>`, and unpacks the `float4` weights the same way, with `cuda::std::get<0/1/2/3>`.
   2. Returns `cuda::std::expected<float, ErrorCode>` holding `ErrorCode::DivisionByZero` for thread 0, which would divide by its own index.
   3. Otherwise divides the numerator by its index, scales the result, and stores an optional "precision bonus" in a `cuda::std::optional<float>` for every fourth thread. `value_or(0.0f)` folds the threads without a bonus into the same expression as the threads with one.
   4. Returns `cuda::std::expected<float, ErrorCode>` holding `ErrorCode::OutOfRange` if the final value leaves a fixed range, and the value itself otherwise.
5. The results are copied back to the host and the device memory is freed.
6. The first sixteen results are printed to the standard output: the thread index, either the value or the error name, and the optional bonus if present.
7. The host recomputes every result with the same `analyze_record` function and configuration, and compares it with what the device produced. It also counts how many threads succeeded, divided by zero, or went out of range, and checks that both failure paths and the success path actually occurred.
8. A `cuda::std::pair<int, float>` is converted to a host `std::pair<int, float>` and back, and the round trip is checked.
9. The result of every check is printed to the standard output.

## Key APIs and Concepts

- `cuda::std::tuple` and `cuda::std::pair` (`<cuda/std/tuple>`, `<cuda/std/utility>`) are the Standard product types, usable on host and device, and trivially copyable across the host/device boundary, including as kernel launch arguments.
- HIP's built-in vector types, such as `float4`, are given a tuple interface by libhipcxx: `cuda::std::tuple_size`, `cuda::std::tuple_element`, and `cuda::std::get<N>` all work on them, so generic code written against `cuda::std::tuple` also works on vector types. This does **not** extend to structured bindings: `auto [x, y, z, w] = some_float4;` is unsupported with `hipcc`, because of how `HIP_vector_type` is implemented internally (a union member). Element access must go through `cuda::std::get<N>`.
- `cuda::std::expected<T, E>` (`<cuda/std/expected>`) holds either a value of type `T` or an error of type `E`. `has_value()` tells the two cases apart, `operator*`/`value()` accesses the value, and `error()` accesses the error. It is constructed from a value directly, or with `cuda::std::unexpect` followed by the error's constructor arguments. Because device code cannot use exceptions, `expected` is the natural way for a `__device__` function to report a per-thread failure, in place of an out-parameter or a sentinel value.
- `cuda::std::optional<T>` (`<cuda/std/optional>`) holds either a value or nothing. `has_value()` tells the two cases apart, and `value_or(default)` returns the value or a default without a separate branch.
- `cuda::std::pair` provides a host-only converting constructor from `::std::pair` and a host-only conversion operator to `::std::pair`, so it interoperates with code that expects the Standard library's own `std::pair`.

## Demonstrated API Calls

### libhipcxx

- `cuda::std::tuple`
- `cuda::std::make_tuple`
- `cuda::std::get`
- `cuda::std::tuple_size`
- `cuda::std::tuple_element`
- `cuda::std::pair`
- `cuda::std::expected`
- `cuda::std::unexpect`
- `cuda::std::optional`
- `cuda::std::nullopt`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`
- `float4`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
- `make_float4`
