# hipThreads SAXPY Step 2: hipThreads Drop-in Example

## Description

This example ports the CPU SAXPY baseline from step 1 to the AMD GPU using hipThreads as a near drop-in replacement for `std::thread`. The key changes from step 1 are:

- `std::thread` is replaced with `hip::wthread`
- The thread lambda receives the `__device__` annotation
- Host arrays are replaced with GPU-resident memory managed by rocThrust
- `hip::wthread::hardware_concurrency()` replaces `std::thread::hardware_concurrency()`

The work-partitioning logic is otherwise unchanged: each `hip::wthread` processes a contiguous chunk of the element array. This demonstrates how hipThreads enables a minimal, structurally faithful GPU port of existing CPU threading code.

### Application flow

1. Allocate and initialize host vectors `x` (all 1.0) and `y` (all 2.0).
2. Allocate device memory using `thrust::make_unique<float[]>` and copy the host data to the GPU via `thrust::copy`.
3. Sleep briefly to allow any asynchronous GPU initialization to complete before starting the timer.
4. Spawn `hip::wthread::hardware_concurrency()` GPU threads, each processing a contiguous chunk of elements via the same scalar loop as step 1.
5. Join all threads.
6. Copy results back to the host via `thrust::copy`.
7. Validate against a CPU reference value and print elapsed time.

## Key APIs and Concepts

### hipThreads

- `hip::wthread` — GPU thread object analogous to `std::thread`. Accepts a `__device__` lambda and its arguments. The lambda executes as a single wavefront on the GPU.
- `hip::wthread::hardware_concurrency()` — returns the total number of wavefronts available across all compute units on the GPU, analogous to `std::thread::hardware_concurrency()`.
- `hip::wthread::join()` — blocks the host until the GPU thread's work is complete.

### rocThrust

- `thrust::make_unique<float[]>(N)` — allocates N floats in GPU device memory.
- `thrust::copy` — transfers data between host and device memory.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`
- `hip::wthread::hardware_concurrency`
- `hip::wthread::join`

### rocThrust

- `thrust::make_unique`
- `thrust::copy`
