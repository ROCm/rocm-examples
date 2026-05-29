# Optical Flow

## Description

This example implements the **Horn-Schunck variational optical flow** algorithm using HIP. It estimates the motion field (displacement vectors) between two consecutive image frames by minimizing a global energy functional that balances data fidelity and spatial smoothness.

The algorithm operates on a Gaussian image pyramid: flow is computed coarse-to-fine, with each level refining the estimate from the level above. At each pyramid level, image warping aligns the target frame with the source, and a Jacobi iterative solver computes the incremental flow update.

The program computes optical flow on both the CPU (`flowGold`) and GPU (`flowHIP`), compares the results via L1 norm, and writes two `.flo` files (Middlebury format) for inspection.

## Application Flow

1. Load two consecutive frames (`data/frame10.ppm`, `data/frame11.ppm`) as single-channel FP32 images.
2. Build a Gaussian pyramid with `nLevels` levels by repeatedly downscaling with a 4-tap filter.
3. At each pyramid level (coarse to fine):
   - Upscale the flow estimate from the coarser level.
   - Warp the target image toward the source using the current flow estimate.
   - Compute image derivatives (Ix, Iy, Iz) via finite differences.
   - Run `nSolverIters` Jacobi iterations to solve for the incremental flow update.
   - Repeat for `nWarpIters` warping passes.
4. Copy GPU results to host and compare against the CPU reference (L1 norm per pixel).
5. Write `FlowGPU.flo` and `FlowCPU.flo` to the working directory.

## Key APIs and Concepts

| Concept | HIP API |
|---|---|
| Texture objects with bilinear filtering | `hipCreateTextureObject`, `hipTextureObject_t` |
| Pitched 2D texture resource | `hipResourceTypePitch2D`, `hipResourceDesc` |
| Mirror address mode | `hipAddressModeMirror` |
| Normalized texture coordinates | `texDescr.normalizedCoords = true` |
| In-kernel texture fetch | `tex2D<float>(tex, x, y)` |
| Block synchronization | `cg::this_thread_block()`, `cg::sync()` |

### Pitch Alignment Requirement

ROCm requires `pitchInBytes` for `hipResourceTypePitch2D` to be a multiple of **256 bytes** (64 floats × 4 bytes). The `STRIDE_ALIGNMENT` constant in `common.h` is set to `64` to satisfy this constraint. CUDA only requires 128 bytes (32 floats), so porting code that used `StrideAlignment = 32` will fail at texture creation.

## Prerequisites

- A ROCm-capable AMD GPU
- ROCm SDK installed ([installation guide](https://rocm.docs.amd.com/en/latest/index.html) or [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md))

## Building

Set `ROCM_PATH` to your ROCm installation root before building. For a standard system install:

```bash
export ROCM_PATH=/opt/rocm
```

For a Python venv-based install (e.g. TheRock):

```bash
export ROCM_PATH=/path/to/venv/lib/python3.12/site-packages/_rocm_sdk_devel
```

### Make

```bash
cd Applications/optical_flow
make ROCM_PATH=$ROCM_PATH
```

If your ROCm device libraries are not found automatically, pass their path explicitly:

```bash
make ROCM_PATH=$ROCM_PATH \
     CXXFLAGS="--rocm-device-lib-path=$ROCM_PATH/lib/llvm/amdgcn/bitcode"
```

### CMake

CMake 3.28 and later require passing `clang++` directly rather than the `hipcc` wrapper script:

```bash
cd Applications/optical_flow
cmake -B build \
  -DROCM_PATH=$ROCM_PATH \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++
cmake --build build -j$(nproc)
```

## Running

If ROCm is not installed to a standard system path, set `LD_LIBRARY_PATH` so the runtime libraries can be found:

```bash
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

The binary locates the sample frames (`data/frame10.ppm`, `data/frame11.ppm`) relative to the source directory automatically, so it can be run from any working directory:

```bash
# Make build
./applications_optical_flow

# CMake build
./build/applications_optical_flow
```

### Expected Output

```text
HSOpticalFlow Starting...

Using device: <GPU name>
Loading "<source-dir>/data/frame10.ppm" ...
Loading "<source-dir>/data/frame11.ppm" ...
Computing optical flow on CPU...
Computing optical flow on GPU...
L1 error : 0.000xxx
```

The program exits with `EXIT_SUCCESS` when the L1 error between the GPU and CPU results is below `0.05`. Two output files are written to the current working directory:

- `FlowGPU.flo` — GPU optical flow result
- `FlowCPU.flo` — CPU reference result

Both files use the [Middlebury `.flo` format](http://vision.middlebury.edu/flow/code/flow-code/README.txt) and can be visualized with tools such as `flowiz` or the Middlebury flow utilities.

## Key Notes

- `STRIDE_ALIGNMENT` is 64 because `hipResourceTypePitch2D` requires `pitchInBytes` to be a multiple of 256 bytes (4 bytes × 64 floats = 256 bytes).
- CMake 3.28+ does not accept the `hipcc` wrapper as `CMAKE_HIP_COMPILER`. Pass the `clang++` binary from `$ROCM_PATH/lib/llvm/bin/clang++` instead.
- The binary resolves the input image paths relative to the source file location at compile time (via `__FILE__`/`/proc/self/exe`), so no specific working directory is required at runtime.
