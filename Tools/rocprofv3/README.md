# rocprofv3 examples

## Description

This directory contains two example shell scripts which showcase some of
`rocprofv3`'s functionality.

The `basic` example script performs the following basic experiments:

* A runtime trace which tracks most relevant ROCm APIs.
* Same as above, but the output file can be analyzed with
  [Perfetto](https://ui.perfetto.dev).
* A system trace including low-level ROCm APIs which are not part of a runtime
  trace. The output is stored as CSV.

The `advanced` example script shows how to employ more detailed features:

* Filtering for specific kernel names.
* Collecting user-defined PMC counters.
* Manually instrumenting the code with the rocTX API.
* PC sampling.

For more information on the functionality of `rocprofv3` or on how to analyze
the output with interactive tools, please refer to its
[documentation](https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

## Building

The example executables can be built either by calling `make` in this directory or by invoking CMake:

```console
user@machine:rocm-examples/Tools/rocprofv3$ cmake -B build && cmake --build build
```

## Usage

After building the example executables the scripts can be executed without
additional parameters as long as the executables reside in the same directory:

```console
user@machine:rocm-examples/Tools/rocprofv3$ cd build && ./rocprofv3-basic.sh
user@machine:rocm-examples/Tools/rocprofv3$ cd build && ./rocprofv3-advanced.sh
```
