# rocThrust Saxpy Example

## Description

This simple program implements the SAXPY operation (`Y[i] = a * X[i] + Y[i]`) using rocThrust and showcases the usage of the vector and functor templates and of `thrust::fill` and `thrust::transform` operations.

### Application flow

1. Two host arrrays of floats `x` and `y` are instantiated, and their contents are printed to the standard output.
2. Two `thrust::device_vector<float>`s, `X` and `Y`, are instantiated with the corresponding arrays. The contents are copied to the device.
3. The `saxpy_slow` function is invoked next. It uses the most straightforward implementation using a temporary device vector `temp` and two separate transformations, one with multiplies and one with plus. First, the `temp` vector is filled with `a` values, using `thrust::fill`. Then, it is filled by transformed values of `a * X[i]` by `thrust::transform` using the `thrust::multiplies` functor. Last, the device vector `Y` is filled by `temp[i] + Y[i]` by `thrust::transform` using the `thrust::plus` functor.
4. The values of device vector `Y` are printed to the standard output. The `X` and `Y` vectors are destroyed.
5. Two new `thrust::device_vector<float>`s, `X` and `Y`, are instantiated with the corresponding arrays. The contents are copied to the device.
6. The `saxpy_fast` function is invoked. It implements the same operation with a single transformation and represents "best practice". Device vector `Y` is filled by `Y[i] = a * X[i] + Y[i]` by `thrust::transform` using `saxpy_functor`. The functor makes use of Fused Multiply-Add (FMA) operation.
7. The values of device vector `Y` are printed to the standard output. The `X` and `Y` vectors are destroyed.

## Key APIs and Concepts

- rocThrust's device and host vectors implement RAII-style ownership over device and host memory pointers (similarly to `std::vector`). The instances are aware of the requested element count, allocate the required amount of memory, and free it upon destruction. When resized, the memory is reallocated if needed.
- Additionally, using `device_vector` and `host_vector` simplifies the transfers between device and host memory to a copy assignment. Note that iterators over device containers can be used everywhere just like host iterators.
- It is suggested that developers use `device_vector` and `host_vector` instead of explicit invocations to `malloc` and `free` functions.
- Likewise `std::fill`, `thrust::fill(first, last, value)` assigns a prescribed `value` to every element in the range `[first, last)`. It can work both with host and device side iterators and supports sequential and parallel executon policies.
- Like `std::transform`, `thrust::transform` can apply both unary and binary functions on its inputs and fills the output range with resulting values.
- Functors `thrust::binary_function`, `thrust::multiplies` and `thrust::plus` represent binary operations correspondingly of general type, of multiplication, and of addition of their arguments.
- [Fused Multiply-Add (FMA)](https://en.cppreference.com/w/cpp/numeric/math/fma) operation `fma` represents multiplication of the first two arguments followed by addition of the third one to the product. It has the advantage of being faster and more accurate compated to separate multiplication and addition on the hardware that support such an instruction, as it avoids cancellation error in addition (addition inside `fma` operation proceeds with full non-rounded result of multiplication that is twice wider).

## Demonstrated API Calls

### rocThrust

- `thrust::host_vector::host_vector`
- `thrust::host_vector::operator[]`
- `thrust::host_vector::begin()`
- `thrust::host_vector::end()`
- `thrust::device_vector::device_vector`
- `thrust::device_vector::operator[]`
- `thrust::device_vector::begin()`
- `thrust::device_vector::end()`
- `thrust::binary_function`
- `thrust::multiplies`
- `thrust::plus`
- `thrust::fill`
- `thrust::transform`
