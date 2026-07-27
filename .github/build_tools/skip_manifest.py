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

# rocDecode leaf directories. All ten need the same test-only skip: they require
# video test data under $ROCM_PATH/share/rocdecode that CI images don't carry.
# ctest self-guards each on `if(EXISTS ...)`, so ctest key is None; only the
# `make test` path needs the explicit skip.
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
    # --- FFT callbacks: test-only, all channels ---------------------------
    # ROCm/rocm-systems#7263: HIP CLR cannot resolve static device symbols via
    # hipModuleGetGlobal; the default store callback aborts at runtime.
    {
        "ctest": "hipfft_callback",
        "path": "Libraries/hipFFT/callback",
        "scope": ["test"],
        "reason": "ROCm/rocm-systems#7263 static device symbol abort at runtime",
    },
    {
        "ctest": "rocfft_callback",
        "path": "Libraries/rocFFT/callback",
        "scope": ["test"],
        "reason": "ROCm/rocm-systems#7263 static device symbol abort at runtime",
    },
    # --- rocDecode: test-only, all channels, no ctest key -----------------
    *[
        {
            "ctest": None,
            "path": f"Libraries/rocDecode/{d}",
            "scope": ["test"],
            "reason": "requires video test data under $ROCM_PATH/share/rocdecode",
        }
        for d in _ROCDECODE_DIRS
    ],
    # --- Stable-only build skip (channel-scoping demonstrator) ------------
    # NOTE: HIP-Basic/cooperative_groups_prefix_sum does not currently exist on
    # amd-staging (only HIP-Basic/cooperative_groups). This entry is inert until
    # that example lands; it documents the stable-only build-skip mechanism.
    # hip_scan.h is absent from the pinned 7.14 stable image (version skew),
    # so if/when the example is added it will not compile on that image.
    {
        "ctest": None,
        "path": "HIP-Basic/cooperative_groups_prefix_sum",
        "scope": ["build"],
        "channels": ["stable"],
        "reason": "hip_scan.h not present in the pinned 7.14 stable image (version skew)",
    },
]
