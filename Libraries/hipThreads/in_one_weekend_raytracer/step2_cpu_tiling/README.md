# hipThreads Ray Tracing in One Weekend Step 2: CPU Tiling Example

## Description

This example extends the step 1 CPU baseline with a spatial tiling optimization. Instead of assigning contiguous row bands to threads, the 1200×800 image is divided into small 8×4 pixel tiles. Worker threads pull tiles from a shared queue, improving cache locality by keeping each thread's working set small enough to fit in the CPU's data cache.

This optimization is a prerequisite for step 3, where the same tiling structure is used to map GPU threads to tiles.

Output is written as a PPM image to standard output; redirect with `> image.ppm`.

This example is based on the original source code from [Ray Tracing in One Weekend](https://github.com/RayTracingGithub/raytracing.github.io) by Peter Shirley, released under the CC0 1.0 public domain dedication. See `../LICENSE.txt` for details.

### Application flow

1. Construct the scene and configure the camera.
2. Divide the 1200×800 image into 8×4 pixel tiles and push them onto a work queue.
3. Spawn `std::thread::hardware_concurrency()` threads. Each thread pops tiles from the queue until all tiles are consumed.
4. Each thread traces all pixels within its current tile, writing results to the output buffer.
5. Join all threads and output the image as PPM to standard output.
