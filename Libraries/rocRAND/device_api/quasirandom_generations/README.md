# rocRAND Quasi-Random Generations Example (C++)

## Description

This example showcases how to perform **quasi-random numbers generation** from device code via rocRAND's **device API**.

As of the moment of writing, rocRAND's device API offers four classes of quasi-random generators. Given the differences in the API for the Sobol and Scrambled Sobol families of generators, this example shows **two use cases**:

1. Generation with a Sobol generator, for 32-bit numbers.
2. Generation with a Scrambled Sobol generator, for 32-bit numbers.

The distribution used is the **uniform** (the default one), and the results are validated by doing a simple uniformity check.

## Application flow

1. Allocate device output buffers.
2. Allocate states.
3. Get direction vectors.
4. For Scrambled Sobol, get scramble constants.
5. Generate values.
6. Copy outputs from device to host and free device memory.
7. Check uniformity of results.

## Key APIs and Concepts

### rocRAND Quasi-Random Engines

The following quasi-random number generators are available in rocRAND:

- Sobol32.
- Sobol64.
- Scrambled Sobol32.
- Scrambled Sobol64.

Both Sobol and Scramble Sobol generators produce low-discrepancy sequences of numbers of 32 or 64 bits in the range $[0, 2^{32}-1]$ or $[0, 2^{64}-1]$, respectively.

Both of them provide API entries for the initialization of their states and for values generation:
- `rocrand_init`: for initializing the generator state from some `direction vectors` (32 or 64, one for each bit of the generated numbers) skipping `offset` values, and
- `rocrand`,  for generating a uniformly distributed random value in the corresponding range.

rocRAND also provide the host methods `rocrand_get_direction_vectors32` and `rocrand_get_direction_vectors64` to retrieve the precomputed direction vectors for both Sobol and Scrambled Sobol.

Additionally, for Scrambled Sobol, the host methods `rocrand_get_scramble_constants32` and `rocrand_get_scramble_constants64` are available for getting the precomputed scramble constants needed by this generator.

Both Sobol and Scrambled Sobol generators also provide a `skipahead` method for skipping a number $n$ of values from the current generated sequences. It is equivalent to calling `rocrand()` $n$ times, but faster.

### Generation Kernel

This example showcases a simple kernel implementation for generating some sequences of quasi-random values cooperatively among all the threads in the grid. Each thread block generates the values for one dimension, and each thread from a thread block generates as many values of the sequence for that dimension as needed.

## Demonstrated API Calls

### rocRAND

#### Device symbols

- `rocrand`
- `rocrand_init`
- `skipahead`

#### Global Symbols

- `rocrand_state_scrambled_sobol32`
- `rocrand_state_sobol32`

#### Host symbols

- `rocrand_get_direction_vectors32`
- `rocrand_get_scramble_constants32`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `gridDim`
- `threadIdx`

#### Host symbols

- `__global__`
- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
