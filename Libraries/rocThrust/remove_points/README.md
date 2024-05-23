# rocThrust Remove Points Example

## Description

This short program demonstrates the usage of the `thrust` random number generation, host vector, generation, tuple, zip iterator, and conditional removal templates.
It generates a number of random points $(x, y)$ in a unit square $x,y\in[0,1)$ and then removes all of them outside the unit circle, i.e. with $x^2 + y^2 > 1$.

## Key APIs and Concepts

- Thrust provides functionality for random number generation similar to [the STL `<random>` header](https://en.cppreference.com/w/cpp/header/random) (from C++11 and above), like `thrust::default_random_engine`, `thrust::uniform_real_distribution` and so on.
- Thrust's vectors implement RAII-style ownership over device and host memory pointers (similarly to `std::vector`). The instances are aware of the requested element count, allocate the required amount of memory, and free it upon destruction. When resized, the memory is reallocated if needed.
- It is suggested that developers use `host_vector` instead of explicit invocations to `malloc` and `free` functions.
- Tuples are heterogeneous, fixed-size collections of values (up to 10 values in rocThrust). Individual elements of a tuple may be accessed with the `get` function.
- `generate(first, last, gen)` assigns the result of invoking `gen`, a function object that takes no arguments, to each element in the range `[first, last)`. It supports device and host side iterators, as well as sequential and parallel execution policies (it can be invoked as `generate(policy, first, last, gen)`).
- The zip iterator provides the ability to parallel-iterate over several controlled sequences simultaneously. A zip iterator is constructed from a tuple of iterators. Moving the zip iterator moves all the iterators in parallel. Dereferencing the zip iterator returns a tuple that contains the results of dereferencing the individual iterators.
- `remove_if` "removes" every element on which the predicate evaluates to `true` from the range specified by begin and end iterators. All kept elements are moved to the beginning of the range in the same order as in the original sequence, and the end iterator to the range of kept elements is returned. Idiomatic usage of conditional removal is the so-called _erase–remove idiom_ `S.erase(remove_if(S.begin(), S.end(), pred), S.end())`. This idiom cannot be used here because the `zip_iterator` refers to multiple containers.

### Application flow

1. A `thrust::default_random_engine` is instantiated and values are sampled from a uniform distribution between 0 and 1 using `thrust::uniform_real_distribution<float>`.
2. To hold the coordinates of the points, two `thrust::host_vector<float>`s are constructed. Their elements are set one-by-one from a uniform distribution by `generate` and the points are printed to the standard output.
3. Zip iterators are constructed from `begin` and `end` iterators over the coordinate vectors and then passed to the `thrust::remove_if` operation. The operation uses a test `is_outside_circle<float>` to remove all points outside the unit circle and puts all remaining points to the beginning of the range spanned by the zip iterators. `thrust::remove_if` returns an end iterator to the remaining points. The new size for vectors is calculated by finding distance between returned iterator and `begin` iterator and the vectors are resized accordingly.
4. Finally, the remaining points are printed again.

## Demonstrated API Calls

### rocThrust

- `thrust::default_random_engine::default_random_engine`
- `thrust::uniform_real_distribution<RealType>::uniform_real_distribution(RealType, RealType)`
- `thrust::uniform_real_distribution<RealType>::operator()(UniformRandomNumberGenerator)`
- `thrust::host_vector::host_vector`
- `thrust::host_vector::operator[]`
- `thrust::host_vector::resize()`
- `thrust::generate`
- `thrust::make_tuple<T1, T2>::make_tuple(T1, T2)`
- `thrust::get<int>(Tuple)`
- `thrust::make_zip_iterator<IteratorTuple>::make_zip_iterator(IteratorTuple)`
- `thrust::zip_iterator<IteratorTuple>::operator-(thrust::zip_iterator<IteratorTuple>)`
- `thrust::remove_if<ForwardIterator, Predicate>(ForwardIterator, ForwardIterator, Predicate)`
