# hipThreads Ray Tracing in One Weekend Step 1: CPU Baseline Example

## Description

This example implements a Monte Carlo path tracer based on Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/). The scene consists of hundreds of randomly placed spheres with Lambertian, metal, and dielectric materials. The renderer shoots 10 camera rays per pixel, each of which recursively scatters until it reaches the sky or exceeds a maximum depth. The image is 1200×800 pixels and is output in PPM format to standard output.

This is the starting point of the ray tracer porting series, using `std::thread` to parallelize across image rows on the CPU. Steps 2 through 4 progressively optimize and port this code to the GPU using hipThreads.

Output is written as a PPM image to standard output; redirect with `> image.ppm`.

This example is based on the original source code from [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io) by Peter Shirley, released under the CC0 1.0 public domain dedication. See `../LICENSE.txt` for details.

### Application flow

1. Construct the scene: hundreds of random spheres (Lambertian, metal, glass) and three large feature spheres.
2. Configure the camera at a fixed viewpoint with depth-of-field blur.
3. Spawn `std::thread::hardware_concurrency()` threads, each responsible for a horizontal band of rows.
4. Each thread traces 10 samples per pixel across its assigned rows, accumulating color via recursive ray scattering, and writes pixels to a shared output buffer.
5. Join all threads.
6. Output the image as PPM to standard output.
