# ROCgdb examples

## Description

This directory contains two example shell scripts which showcase some of `rocgdb`'s functionality.

The `basic` example script performs the following debugging steps:

* A GPU kernel disassembly
* A GPU kernel stackframe examination
* A query of different system and GPU information

The `advanced` example script shows how to perform a more fine-grained analysis:

* Several ways of examining individual GPU registers and buffers in different address spaces
* Modifying wavefront scheduling for better control of wavefront progression

For more information on the functionality of `ROCgdb`, please refer to its
[documentation](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/) or its
[manual](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ROCgdb/gdb/doc/gdb/index.html).

## Building

The example executables can be built either by calling `make` in this directory
or by invoking CMake:

```console
user@machine:rocm-examples/Tools/ROCgdb$ cmake -B build && cmake --build build
```

## Usage

After building the example executable the scripts can be executed without
additional parameters as long as the executable resides in the same directory:

```console
user@machine:rocm-examples/Tools/ROCgdb$ cd build && ./rocgdb-basic.sh
user@machine:rocm-examples/Tools/ROCgdb$ cd build && ./rocgdb-advanced.sh
```

The scripts dump their output to several log files. These are plain text files
which can be opened in any editor to examine their output.
