# rocRAND Simple Distributions Example (C++)

## Description

This sample illustrates the usage of the rocRAND random number generator library via the host-side C++ API. The usage of the random engines and random distributions offered by rocRAND is showcased. The usage, results and execution time of each algorithm provided by rocRAND is compared to the corresponding standard library equivalent.

### Application flow

1. The example command line application takes optional arguments: the used device index, the random distribution type, the element count of the generated random vector and whether the generated vectors should be printed to the standard output.
2. The arguments are parsed in `parse_args` and the result is printed to the standard output. If the parsing fails due to e.g. malformed input, an exception is raised, the correct usage is printed and the program returns with an error code.
3. The utilized device (GPU) is selected in `set_device`. If the selected device does not exist, an error message is printed to the standard error output and the program returns with an error code. Otherwise the name of the selected device is printed to the standard output.
4. The host and device distribution types are selected in `dispatch_distribution_type` based on the provided command line arguments.
5. Two vectors filled with randomly generated values are produced in `compare_device_and_host_random_number_generation`. One is generated on the device using rocRAND (`generate_random_vector_on_device`) and the other is generated on the host using the standard `<random>` library (`generate_random_vector_on_host`). The runtime of the two functions is measured and printed to the standard output.

### Command line interface

The application provides the following optional command line arguments:

- `--device <device ID>`. Controls which device (GPU) the random number generation runs on. Default value is `0`.
- `--distribution uniform_int|uniform_float|normal|poisson`. Controls the type of the random distribution that is used for the random number generation. Default value is `uniform_int`.
- `--size <size>`. Controls the number of random numbers generated.
- `--print`. If specified, the generated random vectors are written to the standard output.

## Key APIs and Concepts

### rocRAND Engines

rocRAND engines define algorithms that generate sequences of random numbers. Typically an engine maintains an internal state that determines the order and value of all subsequent random numbers produced by the engine. In that sense, an engine lacks true randomness, hence the name pseudo-random number generator (or PRNG). Other engines produce quasi-random sequences, which appear to be equidistributed. An engine can be initialized with a seed value that determines the initial state of the engine. Different engine types employ different algorithms to generate the pseudo-random sequence, they differ in the mathematical characteristics of the sequence generated. Unless special requirements arise, it is safe to use the `rocrand_cpp::default_random_engine` alias to create an engine. For the full list of implemented engines, refer to the documentation.

### rocRAND Distributions

A PRNG engine typically generates uniformly distributed integral numbers over the full range of the type. In order to transform this output to something more useful, rocRAND provides a set of distributions that transform this raw random sequence to samples of a random distribution. This example showcases the following distributions:

- `rocrand_cpp::uniform_int_distribution` generates unsigned integers sampled from a [discrete uniform distribution](https://en.wikipedia.org/wiki/discrete_uniform_distribution)
- `rocrand_cpp::uniform_real_distribution` generates floating point numbers sampled from a [continuous uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) over the interval of `[0,1)`
- `rocrand_cpp::normal_distribution` generates floating point numbers sampled from a standard [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).
- `rocrand_cpp::poisson_distribution` generates integers sampled from a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution).

For the full list of implemented distributions, refer to the documentation.

## Demonstrated API Calls

### rocRAND

- `rocrand_cpp::default_random_engine`
- `rocrand_cpp::uniform_int_distribution`
- `rocrand_cpp::uniform_real_distribution`
- `rocrand_cpp::normal_distribution`
- `rocrand_cpp::poisson_distribution`

### HIP runtime

- `hipGetErrorString`
- `hipSetDevice`
- `hipGetDeviceProperties`
- `hipMalloc`
- `hipMemcpy`
- `hipFree`
