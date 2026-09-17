# hipThreads Ray Tracing in One Weekend Step 4: Wavefront SIMD Example

## Description

This example extends the step 3 hipThreads GPU port with wavefront-level SIMD vectorization. Each `hip::wthread` is created with `hip::wthread::max_width()` to fill a full wavefront (typically 64 fibers on AMD CDNA/RDNA hardware). Within each thread, the 64 fibers are each assigned to a single pixel within an 8×4 tile, so all 32 pixels in a tile are traced simultaneously in SIMD lock-step.

The structural change from step 3 is minimal: the thread width is set to `max_width()` and each fiber identifies its pixel via `get_fiber_id()`.

Output is written as a PPM image to standard output; redirect with `> image.ppm`.

This example is based on the original source code from [Ray Tracing in One Weekend](https://github.com/RayTracingGithub/raytracing.github.io) by Peter Shirley, released under the CC0 1.0 public domain dedication. See `../LICENSE.txt` for details.

### Application flow

1. Allocate GPU-resident output buffer and scene.
2. Construct the scene on the GPU via a single `hip::wthread`.
3. Spawn 4096 `hip::wthread`s, each with `hip::wthread::max_width()` fibers.
4. Each fiber identifies its tile and pixel from `get_fiber_id()`, traces its ray, and writes the result to the output buffer.
5. Join all threads, copy the output buffer to the host, and write the PPM image to standard output.

## Key APIs and Concepts

### hipThreads

- `hip::wthread::max_width()` — requests a thread width equal to the GPU's native wavefront size, filling all SIMD lanes. In the ray tracer, 32 fibers cover all 32 pixels of one 8×4 tile simultaneously.
- `hip::this_thread::get_fiber_id()` — returns the 0-based fiber index within the wavefront. In this example it maps to a specific pixel column/row offset within the tile.
- `hip::this_thread::get_width()` — returns the total number of fibers (wavefront width).

### rocThrust

- `thrust::make_unique<T>()` — allocates GPU-resident memory.
- `thrust::copy` — transfers results from GPU memory to the host.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`
- `hip::wthread::max_width`
- `hip::wthread::join`
- `hip::this_thread::get_fiber_id`
- `hip::this_thread::get_width`

### rocThrust

- `thrust::make_unique`
- `thrust::copy`
