# hipThreads SAXPY Step 3: Wavefront SIMD Example

## Description

This example extends the step 2 hipThreads GPU port with wavefront-level SIMD vectorization. Each `hip::wthread` is created with `hip::wthread::max_width()` to request a full wavefront width (typically 64 fibers on AMD CDNA/RDNA hardware). Within each thread, the 64 fibers stride through the element chunk in parallel instead of processing elements sequentially, exposing far more instruction-level parallelism to the GPU's SIMD units.

The structural change from step 2 is minimal: the loop starting index changes from `0` to `hip::this_thread::get_fiber_id()`, and the loop stride changes from `1` to `hip::this_thread::get_width()`.

### Application flow

1. Allocate and initialize host vectors `x` (all 1.0) and `y` (all 2.0).
2. Allocate device memory and copy host data to the GPU via rocThrust.
3. Sleep briefly to allow GPU initialization before starting the timer.
4. Spawn `hip::wthread::hardware_concurrency()` GPU threads, each with `hip::wthread::max_width()` fibers.
5. Each fiber processes elements at stride `get_width()` starting from `get_fiber_id()`, so all fibers in a wavefront execute in lock-step across adjacent elements.
6. Join all threads.
7. Copy results back to the host, validate, and print elapsed time.

## Key APIs and Concepts

### hipThreads

- `hip::wthread::max_width()` — requests a thread width equal to the GPU's native wavefront size, filling all SIMD lanes.
- `hip::this_thread::get_fiber_id()` — returns the 0-based index of the calling fiber within its thread, analogous to a lane ID within a wavefront.
- `hip::this_thread::get_width()` — returns the total number of fibers in the calling thread (the wavefront width).

A fiber is a single SIMD lane within a `hip::wthread`. Creating a thread with width W causes W fibers to execute the lambda concurrently in lock-step, each with a distinct `get_fiber_id()` value. This maps directly to the GPU hardware's wavefront execution model.

### rocThrust

- `thrust::make_unique<float[]>(N)` — allocates N floats in GPU device memory.
- `thrust::copy` — transfers data between host and device memory.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`
- `hip::wthread::hardware_concurrency`
- `hip::wthread::max_width`
- `hip::wthread::join`
- `hip::this_thread::get_fiber_id`
- `hip::this_thread::get_width`

### rocThrust

- `thrust::make_unique`
- `thrust::copy`
