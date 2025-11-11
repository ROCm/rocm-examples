# ROCm Compute Profiler examples

## Description

This directory contains two example shell scripts which showcase some of `rocprof-compute`'s functionality.

The `basic` example script performs the following experiments and analysis steps:

* A full profile
* A reduced profile for roofline analysis
* A full profile where the raw output data is stored in the `*.rocpd` database format
* A system Speed-of-Light analysis using the command line interface (CLI)
* A memory chart analysis using the CLI
* A roofline analysis using the CLI

The `advanced` example script shows how to perform a more fine-grained analysis:

* Applying a substring filter to only profile kernels with a matching name
* Applying a metrics filter to only collect wavefront launch statistics
* Profiling multiple runs and using the CLI to compare the results
* Profiling with program counter (PC) sampling.

For more information on the functionality of `rocprof-compute` or on how to analyze the output with interactive tools,
please refer to its [documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

## Building

The example executables can be built either by calling `make` in this directory
or by invoking CMake:

```console
user@machine:rocm-examples/Tools/rocprof-compute$ cmake -B build && cmake --build build
```

## Usage

After building the example executable the scripts can be executed without
additional parameters as long as the executable resides in the same directory:

```console
user@machine:rocm-examples/Tools/rocprof-compute$ cd build && ./rocprof-compute-basic.sh
user@machine:rocm-examples/Tools/rocprof-compute$ cd build && ./rocprof-compute-advanced.sh
```
