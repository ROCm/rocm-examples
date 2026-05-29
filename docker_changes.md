# Docker Testing Setup — Changes

## Overview

This documents changes made to support building and testing `rocm-examples` inside
a Docker container using the ROCm multi-arch pip install (TheRock nightlies on
Ubuntu 24.04).

---

## New Files

### `~/docker-stuff/Dockerfile`

Ubuntu 24.04 image that installs ROCm via the multi-arch pip wheel
(`https://rocm.nightlies.amd.com/whl-multi-arch/`) and sets up all environment
variables required to build and run `rocm-examples`.

Key design decisions:
- ROCm installed into a Python venv (`/opt/rocm-venv`) via `pip install rocm[libraries,devel,device-all]` followed by `rocm-sdk init`.
- `ROCM_PATH` points at `_rocm_sdk_devel` (headers, compiler, linker stubs).
- `LD_LIBRARY_PATH` orders `_rocm_sdk_libraries/lib` before `_rocm_sdk_devel/lib`
  so full libraries (with kpack archives) are loaded instead of the ~41 KB stubs.
- Render group GID is parameterised (`ARG RENDER_GID=109`) and must match the
  host's render group so `/dev/dri/renderD*` is accessible inside the container.
- A non-root `developer` user is created and added to `video` and `render` groups.

### `~/docker-stuff/run-tests.sh`

Shell script that:
1. Auto-detects the host render group GID.
2. Builds the Docker image (passing `RENDER_GID`).
3. Mounts the `rocm-examples` source tree read-only at `/workspace/rocm-examples`.
4. Runs CMake configure → build → `ctest` inside the container.

Usage:
```bash
./run-tests.sh [/path/to/rocm-examples] [ctest-filter]
# Examples:
./run-tests.sh                            # run all tests
./run-tests.sh ~/rocm-examples
./run-tests.sh ~/rocm-examples 'hip_.*'  # only HIP tests
```

---

## Code Fix

### `HIP-Basic/assembly_to_executable/CMakeLists.txt`

**Problem:** The `add_custom_command` blocks that generate and assemble per-arch
`.s` files were writing their outputs to `CMAKE_CURRENT_SOURCE_DIR` instead of
`CMAKE_CURRENT_BINARY_DIR`. This violates standard CMake practice (generated
artifacts belong in the build tree, not the source tree) and causes a hard build
failure whenever the source directory is not writable — including any read-only
Docker mount, network filesystem, or system-installed source tree.

Error observed inside the container:
```
error: unable to open output file
  '/workspace/rocm-examples/HIP-Basic/assembly_to_executable/main_gfx1100.s':
  'Operation not permitted'
ninja: build stopped: subcommand failed.
```

**Fix:** Changed the `OUTPUT`, `-o` argument, and `DEPENDS` in both `foreach` loops
from `CMAKE_CURRENT_SOURCE_DIR` to `CMAKE_CURRENT_BINARY_DIR`:

```cmake
# Before (generate step)
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s
...
${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s

# After
OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.s
...
${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.s
```

```cmake
# Before (assemble step)
${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s -o
DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s

# After
${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.s -o
DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.s
```

The pre-committed `.s` files in the source tree are now unused by the build;
fresh ones are always generated in the build directory.
