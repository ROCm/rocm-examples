#!/usr/bin/env python3
"""Single source of truth for rocm-examples CI skips.

Every skip in the repo — whether it applies to the ctest run, the `make test`
run, the CMake build, or the `make` build — is one entry in ``SKIP_MANIFEST``.
The generator (``generate_skip_tests.py``) reads this list and produces the
per-consumer artifacts:

  * ``skip_tests.txt``  -> consumed by ``ctest --exclude-from-file``
  * ``skip_build.txt``  -> consumed by ``Common/SkipExamples.cmake``
  * ``SKIP_FROM_TEST``  -> passed on the ``make test`` command line
  * ``SKIP_FROM_BUILD`` -> passed on the ``make`` command line

Entry fields
------------
ctest    : str | None
    The leaf ``example_name`` (globally unique CMake target / ctest name, set at
    ``<leaf>/CMakeLists.txt`` line 23, e.g. ``rocfft_callback``). Used ONLY by
    ctest: it is what goes into ``skip_tests.txt`` for ``ctest
    --exclude-from-file``. ``None`` when the example registers no ctest test, or
    when ctest already self-guards the test (rocDecode guards on test-data
    existence via ``if(EXISTS ...)``) -- then there is nothing for ctest to skip.
path     : str
    Repo-root-relative path to the leaf (e.g. ``Libraries/hipFFT/callback``).
    Used by everything EXCEPT ctest: the ``make`` skip (``SKIP_FROM_*``, matched
    per-directory in the Makefiles) and the CMake ``add_subdirectory`` override
    (exact match). It disambiguates the three ``callback`` directories so a skip
    never hits the wrong one (e.g. rocProfiler-SDK's callback stays built).
scope    : list[str]
    Subset of {"build", "test"}. "build" removes the example from compilation
    (CMake + make); "test" removes it only from the test run (ctest + make test).
reason   : str
    Human-readable justification (shown in the CI step summary).

Optional filters (absent = applies everywhere)
----------------------------------------------
channels : list[str]  -- subset of {"stable", "nightly"}. "stable" = the pinned
    native workflows; "nightly" = the TheRock multi-arch reusable workflow. Use
    this to scope a skip to only one CI channel.
targets  : list[str]  -- match against the --target value (e.g. "gfx1100").
distros  : list[str]  -- match against the --distro value (e.g. "ubuntu-24.04").
install_methods : list[str]  -- match against the --install-method value (e.g.
    "whl-multi-arch", "tarball-multi-arch", "preinstalled"). Use this to scope a
    skip to a specific packaging: whl and tarball are both the "nightly" channel
    but ship different payloads, so this axis is orthogonal to ``channels``.

How to add a skip
-----------------
``path`` is essentially always required -- it drives make (both build and test)
and the CMake build. ``ctest`` is only added on top when you are skipping a test
that the ctest run actually registers.

A BUILD skip implies a TEST skip everywhere -- scope = ["build"] is enough.
On the ctest side the CMake override makes add_test never register. On the make
side the `test:` target filters out SKIP_FROM_BUILD in addition to
SKIP_FROM_TEST (an example that isn't built can't be tested), so `make test`
won't try to rebuild+run a build-skipped example. You only need scope "test"
when you want to skip a test WITHOUT skipping its build (the example compiles
fine but the test itself must not run).

Pick the row that matches what you want:

  * Skip the BUILD (example won't compile on this image/target):
        scope = ["build"], set ``path``, leave ``ctest`` = None.
        -> CMake override + `make` (SKIP_FROM_BUILD) drop it from the build;
           ctest skips it implicitly (add_test never registers) and `make test`
           skips it too (its `test:` target also filters SKIP_FROM_BUILD).

  * Skip only the TEST, and the test IS registered in ctest (runs and fails):
        scope = ["test"], set ``path`` AND ``ctest``.
        -> ctest --exclude-from-file (via ctest) + `make test` (via path).

  * Skip only the TEST, but CMake self-guards add_test (e.g. `if(EXISTS ...)`,
    like rocDecode) so ctest never sees it:
        scope = ["test"], set ``path``, leave ``ctest`` = None.
        -> only `make test` needs skipping (via path); ctest has nothing to skip.

Then optionally narrow with channels / targets / distros (absent =
applies everywhere). Always include a ``reason``.
"""

# rocDecode leaf directories. All ten need the video test data + utility sources
# under $ROCM_PATH/share/rocdecode. The pinned stable image ships these via the
# amdrocm-decode-test package, and the nightly "-tests" tarball carries them too, so
# rocDecode builds and its tests run in both. The nightly whl install does NOT
# carry the data, so the make-test skip is scoped to that install method.
# ctest self-guards each on `if(EXISTS ...)`, so the ctest key is None (ctest
# auto-skips where the data is absent); only the `make test` path needs the
# explicit, install-method-scoped skip.
_ROCDECODE_DIRS = [
    "rocdec_decode",
    "video_decode",
    "video_decode_batch",
    "video_decode_mem",
    "video_decode_multi_files",
    "video_decode_perf",
    "video_decode_pic_files",
    "video_decode_raw",
    "video_decode_rgb",
    "video_to_sequence",
]

SKIP_MANIFEST = [
    # --- rocDecode: test-only, nightly whl install only, no ctest key -----
    # The stable image (amdrocm-decode-test) and the nightly tarball carry the
    # video data, so their tests run; the nightly whl install doesn't, so skip
    # make test only there.
    *[
        {
            "ctest": None,
            "path": f"Libraries/rocDecode/{d}",
            "scope": ["test"],
            "channels": ["nightly"],
            "install_methods": ["whl-multi-arch"],
            "reason": "video test data absent from the TheRock nightly whl install (present on the stable image via amdrocm-decode-test and in the nightly tests tarball)",
        }
        for d in _ROCDECODE_DIRS
    ],
    # --- hipThreads: test-only, gfx1151 only, hipthreads_in_one_weekend_raytracer_step3_hipthread_dropin and hipthreads_in_one_weekend_raytracer_step4_simdize ---
    # oem kernel driver does not have ROCm/amdgpu@55ff0278dd12. This fix needs to be upstreamed for APUs that don’t use dkms to work properly with hipthread.
    # The test failure is due to the driver writing the page tables from the CPU (the default on APUs). The GPU keeps using the address translation it cached in iteration 1,
    # and every later copy reads and writes iteration 1's buffer, which was already freed. ROCm's amdgpu-dkms driver avoids this with a TLB flush after each remap (ROCm/amdgpu@55ff0278dd12),
    # which never reached upstream Linux, so it passes on dGPUs (either using SDMA (page tables are updated on device) or has resizable bar enabled (CPU code path) but with dkms which has the proper fix) and fails on gfx1151 and likely every other APUs without dkms.
    {
        "ctest": "hipthreads_in_one_weekend_raytracer_step3_hipthread_dropin",
        "path": "Libraries/hipThreads/in_one_weekend_raytracer/step3_hipthread_dropin",
        "scope": ["test"],
        "targets": ["gfx1151"],
        "reason": "https://github.com/ROCm/amdgpu/commit/55ff0278dd1239b0bb379d7e0dffa0b16aca6306 is needed for APUs but not available in upstream oem kernel",
    },
    {
        "ctest": "hipthreads_in_one_weekend_raytracer_step4_simdize",
        "path": "Libraries/hipThreads/in_one_weekend_raytracer/step4_simdize",
        "scope": ["test"],
        "targets": ["gfx1151"],
        "reason": "https://github.com/ROCm/amdgpu/commit/55ff0278dd1239b0bb379d7e0dffa0b16aca6306 is needed for APUs but not available in upstream oem kernel",
    },
]
