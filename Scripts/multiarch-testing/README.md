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
Scripts/multiarch-testing/run-tests.sh

# Explicit source path
Scripts/multiarch-testing/run-tests.sh ~/rocm-examples

# Run only tests matching a filter
Scripts/multiarch-testing/run-tests.sh ~/rocm-examples 'hip_.*'
```

Or from within this directory:

```bash
cd Scripts/multiarch-testing
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
  Scripts/multiarch-testing/
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

### Issue 2 — rocBLAS `blas_lib_gfx1100.kpack` near-empty (affects rocSparse and hipSparse too)

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

**rocSparse and hipSparse share this same root cause.** `librocsparse.so` embeds
the kpack path `../.kpack/blas_lib_@GFXARCH@.kpack` — it loads the rocBLAS kpack
for its own GPU kernels rather than a separate sparse kpack. When
`blas_lib_gfx1100.kpack` is near-empty, `hipLaunchKernel` returns
`hipErrorInvalidImage` for every rocsparse kernel launch. `libhipsparse.so` is a
thin wrapper over librocsparse and inherits the same failure. There is no separate
`sparse_lib_*.kpack` and no separate issue to file.

**Tests affected:** all `rocblas_*`, `hipblas_*`, `hipsolver_*`, `rocsolver_*`,
`rocsparse_*`, `hipsparse_*` (~65 tests)

**Error signatures:**
```
rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat:
  Illegal seek for GPU arch : gfx1100

rocSPARSE error encountered: "rocsparse_status_internal_error"
  (underlying: hipLaunchKernel returned hipErrorInvalidImage)
```

---

### Issue 3 — FFT / rocFFT callback kpack near-empty (all arches)

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
