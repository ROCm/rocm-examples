# rocWMMA type-trait regression

This host-only regression checks that `fill_rand` generates negative values for
all four FP8/BF8 types and XF32, with native signed and unsigned controls. Five
columns guarantee the expected values regardless of the random seed. No GPU is
needed to run it; the build needs a HIP SDK and rocWMMA headers.

Build this directory with CMake, then run `ctest --test-dir <build>`; or use
`make test HIP_ARCHITECTURES=gfx1100` with an SDK-supported architecture. It also
participates in the parent rocWMMA CMake and Make builds.

The helper uses `rocwmma::is_signed`, which works with both old and new rocWMMA.
To test the migration with rocWMMA 3.0 or 4.0, add
`-DROCWMMA_LEGACY_STD_TRAIT_SPECIALIZATIONS=0` to `CMAKE_HIP_FLAGS` or `CPPFLAGS`.
The example builds do not impose a compatibility mode on users.
