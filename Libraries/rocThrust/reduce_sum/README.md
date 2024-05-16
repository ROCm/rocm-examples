# rocThrust sum (reduce) example

## Description

An example is presented to compute the sum of a `thrust::device_vector` integer vector using the `thrust::reduce()` generalized summation and the `thrust::plus` operator. The result is written to the standard output.

### Application flow

1. Instantiate a `thrust::host_vector` and fill the elements. The values of the elements are printed to the standard output.
2. Copy the vector to the device by `thrust::device_vector`.
3. Set the initial value of the reduction.
4. Use the `thrust::reduce()` generalized summary function with the `thrust::plus` addition operator and return the sum of the vector.
5. Print the sum to the standard output.

## Key APIs and Concepts

- The `thrust::reduce()` function returns a generalized sum. The summation operator has to be provided by the caller.
- In the example, the operator is the `thrust::plus` function object with integers. It is a binary operator that returns the arithmetic sum.
- A `thrust::device_vector` and a `thrust::host_vector` are used to simplify memory management and transfer. For further details, please visit the [vectors example](../vectors/).

## Demonstrated API Calls

### rocThrust

- `thrust::host_vector::host_vector`
- `thrust::host_vector::operator[]`
- `thrust::device_vector::device_vector`
- `thrust::plus::plus`
- `thrust::reduce()`
