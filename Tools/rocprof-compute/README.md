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

For more information on the functionality of `rocprof-compute`, please refer to its
[documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).