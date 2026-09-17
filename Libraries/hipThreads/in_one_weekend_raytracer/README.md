# hipThreads Ray Tracing in One Weekend Examples

## Description

This series implements a Monte Carlo path tracer based on Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/). The scene consists of hundreds of randomly placed spheres with Lambertian, metal, and dielectric materials. The renderer shoots 10 camera rays per pixel, each of which recursively scatters until it reaches the sky or exceeds a maximum depth. The image is 1200×800 pixels and is output in PPM format to standard output.

The series starts from a CPU baseline using `std::thread`, introduces a tiling optimization, and then progressively ports the renderer to the AMD GPU using hipThreads.

This series is based on the original source code from [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io) by Peter Shirley, released under the CC0 1.0 public domain dedication. See `LICENSE.txt` for details.

### Steps

| Step | Directory | Description |
|------|-----------|-------------|
| 1 | `step1_baseline/` | CPU baseline: `std::thread` parallelism across image rows. |
| 2 | `step2_cpu_tiling/` | CPU tiling: threads pull 8×4 pixel tiles from a shared work queue for better cache locality. |
| 3 | `step3_hipthread_dropin/` | GPU port: `std::thread` replaced with `hip::wthread` as a near drop-in; scene constructed on the GPU. |
| 4 | `step4_simdize/` | Wavefront SIMD: each `hip::wthread` is widened to fill a full wavefront, with each fiber tracing one pixel. |
