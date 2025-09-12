# hipRAND Device API Quasirandom Generation Example

## Description

This example illustrates the use of the hipRAND device API. It specifically shows an example for a quasirandom generator inside a kernel.

### Application flow

1. Define generator state types:
    - hiprandStateSobol32 (Sobol).
    - hiprandStateScrambledSobol32 (Scrambled Sobol).
2. Allocate device output buffers
3. Allocate generator states (one per thread).
4. Get direction vectors
5. Get scramble constants (Scrambled Sobol only).
6. Generate Sobol values
    - Launch sobol_init_kernel to initialize Sobol states.
    - Launch generate_kernel to produce Sobol random numbers.
7. Generate Scrambled Sobol values
    - Launch scrambled_sobol_init_kernel to initialize states.
    - Launch generate_kernel to produce Scrambled Sobol random numbers.
8. Copy results back to host and free device memory
9. Free all allocated device memory (outputs, states, vectors, scramble constants).
10. Validate results
    - Check uniformity of Sobol-generated numbers.
    - Check uniformity of Scrambled Sobol-generated numbers.
    - Print validation results (success/failure).

## Key APIs and Concepts

### hipRAND

- The device-level hipRAND API is used to create quasirandom number generators directly on the GPU, here with hiprandStateSobol32 and hiprandStateScrambledSobol32.
- Sobol generators require direction vectors, which are retrieved using hiprandGetDirectionVectors32 and copied to the device for use in state initialization.
- Scrambled Sobol generators also require scramble constants, provided by hiprandGetScrambleConstants32, to apply an additional randomization to the Sobol sequence.

## Used API surface

### hipRAND

- `hiprand_init`
- `hiprand`
- `hiprandStateSobol32`
- `hiprandStateScrambledSobol32`
- `hiprandDirectionVectors32_t`
- `hiprandGetDirectionVectors32`
- `HIPRAND_DIRECTION_VECTORS_32_JOEKUO6`
- `HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`
- `hiprandGetScrambleConstants32`

### HIP runtime

- `HIP_CHECK`
- `hipMalloc`
- `hipDeviceSynchronize`
- `hipGetLastError`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipFree`
