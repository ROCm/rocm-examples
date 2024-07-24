# rocRAND Pseudo-Random Generations Example (C++)

## Description

This example showcases how to perform **pseudo-random numbers generation** from device code via rocRAND's **device API**.

As of the moment of writing, rocRAND's device API offers six classes of pseudo-random generators. Given the differences in the API for the MTGP32 and LFSR113 generators, this example shows **three use cases**:

1. Generation with a pseudo-random generator (other than MTGP32 and LFSR113), which in this case is XORWOW.
2. Generation with MTGP32.
3. Generation with LFSR113.

The distribution used is the **uniform** (the default one), and the results are validated by doing a simple uniformity check.

## Application flow

1. Allocate device output buffers.
2. For MTGP32, allocate states (as many as thread blocks in the grid).
3. Generate values.
4. Copy outputs from device to host and free device memory.
5. Check uniformity of results.

## Key APIs and Concepts

### rocRAND Pseudo-Random Engines

The following pseudo-random number generators are available in rocRAND:

- XORWOW.
- MRG32K3A.
- MTGP32.
- Philox 4x32-10.
- MRG31K3P.
- LFSR113.
- MT19937.
- ThreeFry 2x32-20, 4x32-30, 2x64-20 and 4x64-20.

ThreeFry 2x64-20 and 4x64-20 produce 64-bit pseudo-random values within the range $[0, 2^{64} - 1]$, while all the others produce 32-bit pseudo-random values within the range $[0, 2^{32} - 1]$.

All of them provide API entries for the initialization of their internal state and for values generation:
- `rocrand_init`: for initializing the generator state from a `seed`, skipping `subsequence` subsequences and `offset` values, and
- `rocrand`,  for generating a uniformly distributed random value in the corresponding range.

However, for MTGP32 and LFSR113 these interfaces change, due to the requirements of these generators:

- **MTGP32** generates values for multiple sequences, and each of said sequences is kept in a `rocrand_state_mtgp32` state structure. Thus, MTGP32 provides methods for intializing and copying these states in an efficient way by utilizing all threads in a thread block:
  
  - `rocrand_make_state_mtgp32`: for initializing a whole array of `rocrand_state_mtgp32` states, and
  - `rocrand_mtgp32_block_copy`: for copying one `rocrand_state_mtgp32` to another. This method may be used to load/write one state from/to global memory before/after a block generates values for it.

- **LFSR113** is a combined LFSR generator that uses four LFSR recurrences to produce its random numbers. Because of this, for its initialization it needs four seeds (instead of one like other pseudo-random generators). Therefore, `rocrand_init` requires a `uint4` seed value. Another particularity of the LFSR113 generator is that it doesn't support an `offset` parameter for skipping a certain amount of values during the initialization of the state with `rocrand_init`.

Some other API entries available are:

- `skipahead`: For skipping a number $n$ of values from the current generated sequence. It is equivalent to calling `rocrand()` $n$ times, but faster.
- `skipahead_sequence`: For skipping a number $n$ of sequences (of length $m$, varying from one generator to another). It is equivalent to calling `rocrand()` $n \times m$ times, but faster.
- `skipahead_subsequence`: For skipping a number $n$ of subsequences (of length $m$, varying from one generator to another). It is equivalent to calling `rocrand()` $n \times m$ times, but faster.

These are present for MRG31K3P, MRG32K3A, Philox 4X32-10 and XORWOW.

### Generation Kernel

This example showcases a simple kernel implementation for generating a sequence of pseudo-random values cooperatively among all the threads in the grid. Each thread generates as many values of the sequence as needed, with a stride equal to the size of the grid (in number of threads). For generating a number of values that is a multiple of the size of the grid, the load among the threads will be balanced as each of them will generate the same amount of values. On the other hand, when the size is not a multiple of the total amount of threads available, some threads will generate one more value than others.

Separate kernels are provided for MTGP32 and LFSR113 because of their APIs particularities. 
For MTGP32, the kernel implemented assumes that each thread block is going to generate values for only one of the sequences. For LFSR113, the initialization call uses a 4-dimensional seed with default seeds for each LFSR sequence.

## Demonstrated API Calls

### rocRAND

#### Device symbols

- `rocrand`
- `rocrand_mtgp32_block_copy`
- `rocrand_init`

#### Global Symbols

- `rocrand_state_lfsr113`
- `rocrand_state_mtgp32`
- `rocrand_state_xorwow`

#### Host symbols

- `rocrand_make_state_mtgp32`

### HIP runtime

#### Device symbols

- `__shared__`
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
