# rocThrust Norm Example

## Description

An example is presented to compute the Euclidean norm of a `thrust::device_vector`. The result is written to the standard output.

### Application flow

1. Instantiate a host vector.
2. Copy the vector to the device by constructing `thrust::device_vector` from the host vector.
3. Set the initial value for the transformed reduction to 0.
4. Add the sum of the square of each element and get the square root of the sum (by using `std::sqrt()`). That is the definition of the Euclidean norm. Use the `square` operator to calculate the square of each element. Use the `thrust::plus` binary operator to sum elements.
5. Print the norm to the standard output.

## Key APIs and Concepts

- `thrust::transform_reduce()` computes a generalized sum (AKA reduction or fold) after transforming each element with a unary function. Both the transformation and the reduction function can be specified. (e.g. with `thrust::plus` as the binary summation and `f` as the transform function `transform_reduce` would compute the value of `f(a[0]) + f(a[1]) + f(a[2]) + ...`).
- In the example, the operator is the `thrust::plus` function object with doubles. It is a binary operator that returns the arithmetic sum.
- An initial value is required for the summation.
- A `thrust::device_vector` is used to simplify memory management and transfer. See the [vectors example](../vectors) for the usage of `thrust::vector`.

## Demonstrated API Calls

### rocThrust

- `thrust::device_vector::device_vector`
- `thrust::plus`
- `thrust::reduce()`
