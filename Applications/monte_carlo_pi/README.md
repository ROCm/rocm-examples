# Applications Monte Carlo Pi Example

## Description

This example demonstrates how the mathematical constant pi ($\pi$) can be approximated using Monte Carlo integration. Monte Carlo integration approximates integration of a function by generating random values over a domain that is the superset of the function's domain. Using the ratio between the number of samples in both domains and the range of the random values, the integral is approximated.

The area of a disk is given by $r^2\pi$, where $r$ is the radius of a disk. Uniform random values are typically generated in the range $(0,1]$. Using a disk of radius $1$ centered on the origin, a sample point is in the disk if it's distance to the origin is less than $1$. The ratio between the number of sample points within the disk and the total sample points is an approximation of the ratio between the area of the disk and the quadrant $(0,1]\times(0,1]$, which is $\frac{\pi}{4}$. Multiplying the sample point ratio by $4$ approximates the value of pi.

To generate a large number of random samples we use hipRAND, a platform-independent library for GPU-based random number generation. hipRAND offers a choice of different generators, belonging to one of two categories: pseudorandom and quasirandom. Pseudorandom-number generators output a stream of numbers that appears to be statistically random, but is deterministic based on a seed. Quasirandom-number generators output a stream of values that cover the output domain evenly, for each given domain. For Monte Carlo integration, is it assumed that a quasirandom-number generator will provide a better approximation with a low number of points, because the sample points are of a guaranteed statistical distribution.

To compute the number of sample points that lie within the disk, we use hipCUB, which is a platform-independent library providing GPU primitives. For each sample, we are looking to compute whether it lies in the disk, and to count the number of samples for which this is the case. Using and indicator function and `TransformInputIterator`, an iterator is created which outputs a zero or one for each sample. Using `DeviceReduce::Sum`, the sum over the iterator's values is computed.

### Application flow

1. Parse and validate user input.
2. Allocate device memory to store the random values. Since the samples are two-dimensional, two random values are
   required per sample.
3. Initialize hipRAND's default pseudorandom-number generator and generate the required number of values.
4. Allocate and initialize the input and output for hipCUB's `DeviceReduce::Sum`:

   a) Create a `hipcub::CountingInputIterator` that starts from `0`, which will represent the sample index.

   b) Create a `hipcub::TransformInputIterator` that uses the sample index to obtain the sample's coordinates from the
       array of random numbers, and computes whether it lies within the disk. This iterator will be the input for the
       device function.

   c) Allocate device memory for the variable that stores the output of the function.

5. Calculate the required amount of temporary storage, and allocate it.
6. Calculate the number of samples within the disk with `hipcub::DeviceReduce::Sum`.
7. Copy the result back to the host and calculate pi.
8. Clean up the generator and print the result.

9. Initialize hipRAND's default quasirandom-number generator, set the dimensions to two, and generate the required number of values.

   Note that the first half of the array will be the first dimension, the second half will be the second dimension.

10. Repeat steps 4. - 8. for the quasirandom values.

### Command line interface

- `-s <sample_count>` or `-sample_count <sample_count>` sets the number of samples used, the default is $2^{20}$.

## Key APIs and Concepts

- To start using hipRAND, a call to `hiprandCreateGenerator` with a generator type is made.

  - To pick any of hipRAND's pseudorandom-number generators, we use type `HIPRAND_RNG_PSEUDO_DEFAULT`. For pseudorandom-number generators, the seed can be set with `hiprandSetPseudoRandomGeneratorSeed`.
  - We use type `HIPRAND_RNG_QUASI_DEFAULT` to create a quasirandom-number generator. For quasirandom-number generators, the number of dimensions can be set with `hiprandSetQuasiRandomGeneratorDimensions`. For this example, we calculate an area, so our domain consists of two dimensions.

  Destroying the hipRAND generator is done with `hiprandDestroyGenerator`.

- hipCUB itself requires no initialization, but each of its functions must be called twice. The first call must have a null-valued temporary storage argument, the call sets the required storage size. The second call performs the actual operation with the user-allocated memory.

- hipCUB offers a number of iterators for convenience:

  - `hipcub::CountingInputIterator` will act as an incrementing sequence starting from a specified index.
  - `hipcub::TransformInputIterator` takes an iterator and applies a user-defined function on it.

- hipCUB's `DeviceReduce::Sum` computes the sum over the input iterator and outputs a single value to the output iterator.

## Demonstrated API Calls

### HIP runtime

- `__device__`
- `__forceinline__`
- `__host__`
- `hipError_t`
- `hipEventCreate`
- `hipEventDestroy`
- `hipEventElapsedTime`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamDefault`

### hipRAND

- `HIPRAND_RNG_PSEUDO_DEFAULT`
- `HIPRAND_RNG_QUASI_DEFAULT`
- `HIPRAND_STATUS_SUCCESS`
- `hiprandCreateGenerator`
- `hiprandDestroyGenerator`
- `hiprandGenerateUniform`
- `hiprandGenerator_t`
- `hiprandSetPseudoRandomGeneratorSeed`
- `hiprandSetQuasiRandomGeneratorDimensions`
- `hiprandStatus_t`

### hipCUB

- `hipcub::CountingInputIterator`
- `hipcub::DeviceReduce::Sum`
- `hipcub::TransformInputIterator`
