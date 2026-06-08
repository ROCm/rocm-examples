# Multi-Arch Testing with Docker

## Overview

Docker environment for building and testing `rocm-examples` against the ROCm
multi-arch pip wheel (TheRock nightlies) on Ubuntu 24.04.

## Files

- `Dockerfile` — Ubuntu 24.04 image with ROCm installed via multi-arch pip
- `run-tests.sh` — builds the image and runs the full `ctest` suite with GPU access

## Usage

From the repo root:

```bash
# Run all tests (source dir auto-detected as repo root)
Scripts/MultiArchTesting/run-tests.sh

# Explicit source path
Scripts/MultiArchTesting/run-tests.sh ~/rocm-examples

# Run only tests matching a filter
Scripts/MultiArchTesting/run-tests.sh ~/rocm-examples 'hip_.*'
```

Or from within this directory:

```bash
cd Scripts/MultiArchTesting
./run-tests.sh
./run-tests.sh ~/rocm-examples 'rocblas_.*'
```

The script auto-detects the host render group GID so `/dev/dri/renderD*` is
accessible inside the container. GPU passthrough requires `/dev/kfd` and
`/dev/dri` to be present on the host.

## Dockerfile notes

- ROCm is installed into `/opt/rocm-venv` via `pip install rocm[libraries,devel,device-all]` followed by `rocm-sdk init`.
- `ROCM_PATH` points at `_rocm_sdk_devel` (headers, compiler, linker stubs).
- `LD_LIBRARY_PATH` orders `_rocm_sdk_libraries/lib` before `_rocm_sdk_devel/lib`
  so full libraries (with kpack archives) are loaded instead of the ~41 KB linker stubs.
- Render group GID is parameterised (`ARG RENDER_GID=109`) and passed in by `run-tests.sh`.
- A non-root `developer` user is created and added to `video` and `render` groups.

To target a specific nightly date or a single GPU arch (faster/smaller image):

```bash
docker build \
  --build-arg ROCM_INDEX_URL=https://rocm.nightlies.amd.com/whl-multi-arch/ \
  --build-arg ROCM_EXTRAS=libraries,devel,device-gfx1100 \
  -t rocm-examples-test \
  Scripts/MultiArchTesting/
```

## Known Issues

As of ```rocm==7.14.0a20260605```, all test cases (306/306) pass.
