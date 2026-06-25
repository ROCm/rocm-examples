# hipFile Get Version Example

## Description

This program demonstrates how to query the hipFile library version at both compile time and runtime.
It is the simplest possible hipFile program and a useful first check that the library is installed
and visible to the build system.

### Application flow

1. The compile-time macros `HIPFILE_VERSION_MAJOR`, `HIPFILE_VERSION_MINOR`, and
   `HIPFILE_VERSION_PATCH` are printed. These are defined in `<hipfile.h>` and are resolved by the
   preprocessor before the program runs.
2. `hipFileGetVersion` is called to retrieve the same version numbers at runtime from the loaded
   shared library. This confirms that the runtime library matches the headers used at compile time.
3. Both version strings are printed and the program exits.

## Key APIs and Concepts

- `HIPFILE_VERSION_MAJOR`, `HIPFILE_VERSION_MINOR`, `HIPFILE_VERSION_PATCH` are preprocessor
  constants defined in `<hipfile.h>`. They reflect the version of the headers against which the
  program was compiled. Use them in `#if` guards to conditionally compile code that requires a
  specific hipFile version.
- `hipFileGetVersion` queries the runtime library for its version numbers. Comparing the compile-time
  and runtime versions is good practice when deploying to systems where the installed library version
  may differ from the one used during development.
- `hipFileError_t` is the unified error type returned by most hipFile API calls. It bundles a
  hipFile-specific error code (`err`) with an underlying HIP driver error code (`hip_drv_err`).

## Demonstrated API Calls

### hipFile runtime

- `hipFileGetVersion`

### hipFile preprocessor constants

- `HIPFILE_VERSION_MAJOR`
- `HIPFILE_VERSION_MINOR`
- `HIPFILE_VERSION_PATCH`
