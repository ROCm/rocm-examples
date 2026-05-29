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

---

## Known Upstream Packaging Issues (TheRock multi-arch pip wheel)

Test run on gfx1100 (Navi31 / RX 7900 XTX) against nightly `7.14.0a20260529`
produced **107/306 failures** caused entirely by missing or corrupt per-arch GPU
kernel artifacts in the multi-arch pip wheel. These are upstream TheRock bugs —
no fix is needed in `rocm-examples`.

### Issue 1 — hipBLASLt gfx1100 entirely absent

**TheRock issue:** [#5478](https://github.com/ROCm/TheRock/issues/5478) (open, filed 2026-05-27, no fix in progress)

`TensileLibrary_lazy_gfx1100.dat`, `Kernels.so-000-gfx1100.hsaco`,
`hipblasltTransform_gfx1100.hsaco`, and `extop_gfx1100.co` are all completely
absent from `_rocm_sdk_libraries/lib/hipblaslt/library/` when installing via the
multi-arch pip index. Every peer arch (gfx1101–gfx1103, gfx1200, gfx1201, gfx908,
gfx90a, gfx942, gfx950) is present. The `device_artifact_filter` in TheRock's
`build_python_packages.py` likely does not include hipblaslt, so gfx1100 artifacts
are silently excluded from the device wheel (same class of omission as RCCL before
commit 608407f).

**Tests affected:** all `hipblaslt_*` (38 tests)

**Error signatures:**
```
rocblaslt error: Could not load ".../hipblaslt/library/TensileLibrary_lazy_gfx1100.dat"
hipModuleLoad failed: .../hipblaslt/library/Kernels.so-000-gfx1100.hsaco error: file not found
terminate called after throwing an instance of 'std::runtime_error'
  what():  Unexpected EOF!
```

---

### Issue 2 — rocBLAS `blas_lib_gfx1100.kpack` near-empty

**TheRock issue:** None filed for gfx1100 specifically. Closest precedent:
[#5179](https://github.com/ROCm/TheRock/issues/5179) (open) — same symptom for
gfx942 in a different drop.

`blas_lib_gfx1100.kpack` is **6.8 KB** in the current nightly; every other modern
arch is **~42 MB**. This is the artifact-splitter overwrite bug: a later build job
produces a near-empty kpack and overwrites the correct one during wheel packaging
(same root cause class as `fft_lib_*` and `blas_lib_gfx906` in #5274).
As a result `TensileLibrary_lazy_gfx1100.dat` and `Kernels.so-000-gfx1100.hsaco`
are never extracted into `rocblas/library/`, so rocBLAS cannot load kernels for
gfx1100.

**Tests affected:** all `rocblas_*`, `hipblas_*`, `hipsolver_*`, `rocsolver_*`
(~30 tests)

**Error signature:**
```
rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat:
  Illegal seek for GPU arch : gfx1100
```

---

### Issue 3 — rocSparse / hipSparse failures

**TheRock issue:** None filed.

All `rocsparse_*` and `hipsparse_*` tests fail. The `.kpack/` directory contains
no `sparse_lib_*.kpack` entries for any arch, suggesting the sparse library either
uses a different kernel delivery mechanism that is broken, or is missing from
`device_artifact_filter` entirely.

**Tests affected:** all `rocsparse_*` and `hipsparse_*` (~35 tests)

---

### Issue 4 — FFT / rocFFT callback kpack near-empty (all arches)

**TheRock issue:** Tracked separately from prior investigation (see ROCM-25114).

`fft_lib_gfx1100.kpack` is **2.8 KB**; every other arch is identically 2.7–2.8 KB,
confirming the FFT callback kpack is near-empty across all architectures. Callback
kernels (e.g. `store_cb_default_complex_double`) cannot be loaded because the HSACO
is absent from the kpack payload.

**Tests affected:** `hipfft_callback`, `rocfft_callback`

**Error signature:**
```
Cannot create GlobalVar Obj for symbol:
  _ZL31store_cb_default_complex_double.static.4d22c59b5a98cebc
```
