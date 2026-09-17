# hipThreads Ray Tracing in One Weekend Step 3: hipThreads Drop-in Example

## Description

This example ports the tiled CPU ray tracer from step 2 to the AMD GPU using hipThreads as a near drop-in replacement for `std::thread`. 4096 `hip::wthread`s are spawned, each processing a strided set of 8×4 pixel tiles. The scene is also constructed on the GPU via a dedicated `hip::wthread`.

The key changes from step 2 are:

- `std::thread` is replaced with `hip::wthread`

- Device lambdas receive the `__device__` annotation

- Host memory is replaced with GPU-resident memory managed by rocThrust

- The scene object (`hittable_list`) is allocated and populated on the GPU

Output is written as a PPM image to standard output; redirect with `> image.ppm`.

This example is based on the original source code from [Ray Tracing in One Weekend](https://github.com/RayTracingGithub/raytracing.github.io) by Peter Shirley, released under the CC0 1.0 public domain dedication. See `../LICENSE.txt` for details.

### Application flow

1. Allocate a GPU-resident output buffer and scene using `thrust::unique_ptr`.
2. Spawn a single `hip::wthread` to construct the scene on the GPU (`random_scene()`).
3. Join the construction thread.
4. Spawn 4096 `hip::wthread`s, each processing a strided set of tiles from the full tile list.
5. Each thread traces all pixels within each of its tiles using recursive ray scattering.
6. Join all threads and copy the output buffer back to the host.
7. Output the image as PPM to standard output.

## Key APIs and Concepts

### hipThreads

- `hip::wthread` — GPU thread object. Each thread processes a strided subset of tiles from the 1200×800/32 tile list.

- `hip::wthread::join()` — blocks the host until the GPU thread completes.

### rocThrust

- `thrust::make_unique<T>()` — allocates GPU-resident memory for the pixel buffer and scene objects.

- `thrust::copy` — copies the rendered pixel buffer from GPU memory to the host.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`

- `hip::wthread::join`

### rocThrust

- `thrust::make_unique`

- `thrust::copy`
