# ROCm Systems Profiler examples

## Description

This directory contains two example shell scripts which showcase some of `rocprof-sys`'s functionality.

The `basic` example script performs the following basic experiments:

* Call-stack sampling
* Profiling and tracing
* Runtime instrumentation

The `advanced` example script shows how to instrument an application by rewriting the binary. In addition, the following
example steps are performed:

* Restricting the experiment to the profiling and tracing of HIP API calls
* Collecting additional CPU performance counters during profiling
* Manually instrumenting the code with the ROCm Systems Profiler user API

For more information on the functionality of `rocprof-sys` or on how to analyze the output with interactive tools,
please refer to its [documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## Building

The example executables can be built either by calling `make` in this directory
or by invoking CMake:

```console
$ user@machine:rocm-examples/Tools/rocprof-systems$ cmake -B build && cmake --build build
```

## Usage

After building the example executables the scripts can be executed without
additional parameters as long as the executables reside in the same directory:

```console
user@machine:rocm-examples/Tools/rocprof-systems$ cd build && ./rocprof-systems-basic.sh
user@machine:rocm-examples/Tools/rocprof-systems$ cd build && ./rocprof-systems-advanced.sh
```
