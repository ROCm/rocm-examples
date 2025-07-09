# HIP-Basic Assembly to Executable Example

## Description

This example shows how to manually compile and link a HIP application from device assembly. Pre-generated assembly files are compiled into an _offload bundle_, a bundle of device object files, and then linked with the host object file to produce the final executable.

Building HIP executables from device assembly can be useful for example to experiment with specific instructions, perform specific optimizations, or can help debugging.

### Building

- Build with Makefile: to compile for specific GPU architectures, optionally provide the HIP_ARCHITECTURES variable. Provide the architectures separated by comma.

    ```shell
    make HIP_ARCHITECTURES="gfx803;gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1201"
    ```

- Build with CMake:

    ```shell
    cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES="gfx803;gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1201"
    cmake --build build
    ```

    On Windows the path to RC compiler may be needed: `-DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/path/to/x64/rc.exe"`
    HIP SDK for Windows does not support HIP device architecture gfx942.

## Generating device assembly

This example creates a HIP file from device assembly code, however, such assembly files can also be created from HIP source code using `hipcc`. This can be done by passing `-S` and `--cuda-device-only` to hipcc. The former flag instructs the compiler to generate human-readable assembly instead of machine code, and the latter instruct the compiler to only compile the device part of the program. The assembly files for this example were generated as follows:

```shell
$ROCM_INSTALL_DIR/bin/hipcc -S --cuda-device-only --offload-arch=gfx803 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx950 --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1151 --offload-arch=gfx1201 main.hip
```

The user may modify the `--offload-arch` flag to build for other architectures and choose to either enable or disable extra device code-generation features such as `xnack` or `sram-ecc`, which can be specified as `--offload-arch=<arch>:<feature>+` to enable it or `--offload-arch=<arch>:<feature>-` to disable it. Multiple features may be present, separated by colons.

## Build Process

A HIP binary consists of a regular host executable, which has an offload bundle containing device code embedded inside it. This offload bundle contains object files for each of the target devices that it is compiled for, and is loaded at runtime to provide the machine code for the current device. A HIP executable can be built from device assembly files and host HIP code according to the following process:

1. The `main.hip` file is compiled to an object file that only contains host code with `hipcc` by using the `--cuda-host-only` option. `main.hip` is a program that launches a simple kernel to compute the square of each element of a vector. The `-c` option is required to prevent the compiler from creating an executable, and make it create an object file containing the compiled host code instead.

    ```shell
    $ROCM_INSTALL_DIR/bin/hipcc -c --cuda-host-only main.hip
    ```

2. Each device assembly file is compiled to a device object file using `clang`. This requires specifying the correct architecture using `-target amdgcn-amd-amdhsa`, and the target architecture that should be assembled for using `-mcpu`:

    ```shell
    $ROCM_INSTALL_DIR/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 main_gfx1030.s -o main_gfx1030.o
    $ROCM_INSTALL_DIR/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=<arch> main_<arch>.s -o main_<arch>.o
    ...
    ```

3. The device object files are combined into an offload bundle using `clang-offload-bundler`. This requires specifying the target as well as the offload kind for each device, in the form `<offload-kind>-<target>[-<target id>[:<target features>]]`. For HIP device code, `<offload-kind>` is `hipv4`. Note that this command requires an (empty) entry for the host to also be present, with `<offload-kind>` `host`. The order of targets and inputs must match. `<target>` is an LLVM target 4-tuple, which is specified as `<arch>-<vendor>-<os>-<env>`.

    ```shell
    $ROCM_INSTALL_DIR/llvm/bin/clang-offload-bundler -type=o -bundle-align=4096 \
            -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1030,hipv4-... \
            -input=/dev/null \
            -input=main_gfx1030.o -input=... \
            -output=offload_bundle.hipfb
    ```

    Note: using -bundle-align=4096 only works on ROCm 4.0 and newer compilers. Also, the architecture must match the same `--offload-arch` as when compiling to assembly.

4. The offload bundle is embedded inside an object file that can be linked with the object file containing the host code. The offload bundle must be placed in the `.hip_fatbin` section, and must be placed after the symbol `__hip_fatbin`. This can be done by creating an assembly file that places the offload bundle in the appropriate section using the `.incbin` directive:

    ```nasm
        .type __hip_fatbin,@object
        ; Tell the assembler to place the offload bundle in the appropriate section.
        .section .hip_fatbin,"a",@progbits
        ; Make the symbol that addresses the binary public
        .globl __hip_fatbin
        ; Give the bundle the required alignment
        .p2align 12
    __hip_fatbin:
        ; Include the binary
        .incbin "offload_bundle.hipfb"
    ```

    This file can then be assembled using `llvm-mc` as follows:

    ```shell
    $ROCM_INSTALL_DIR/llvm/bin/llvm-mc -triple <host target> -o main_device.o hip_obj_gen.mcin --filetype=obj
    ```

5. Finally, using the system linker, hipcc, or clang, the host object and device objects are linked into an executable:

    ```shell
    <ROCM_PATH>/hip/bin/hipcc -o hip_assembly_to_executable main.o main_device.o
    ```

### Visual Studio 2019

The above compilation steps are implemented in Visual Studio through Custom Build Steps and Custom Build Tools:

- The host compilation from step 1 is performed by adding extra options to the source file, under `main.hip -> properties -> C/C++ -> Command Line`:

    ```shell
    Additional Options: --cuda-host-only
    ```

- Each device assembly .s file has a custom build tool associated to it, which performs the operation associated to step 2 from the previous section:

    ```shell
    Command Line: "$(ClangToolPath)clang++" -o "$(IntDir)%(FileName).o" "%(Identity)" -target amdgcn-amd-amdhsa -mcpu=gfx90a
    Description: Compiling Device Assembly %(Identity)
    Output: $(IntDir)%(FileName).o
    Execute Before: ClCompile
    ```

- Steps 3 and 4 are implemented using a custom build step:

    ```shell
    Command Line:
        "$(ClangToolPath)clang-offload-bundler" -type=o -bundle-align=4096 -targets=host-x86_64-pc-windows-msvc,hipv4-amdgcn-amd-amdhsa--gfx900,hipv4-amdgcn-amd-amdhsa--gfx906,hipv4-amdgcn-amd-amdhsa--gfx908,hipv4-amdgcn-amd-amdhsa--gfx90a,hipv4-amdgcn-amd-amdhsa--gfx942,hipv4-amdgcn-amd-amdhsa--gfx1030,hipv4-amdgcn-amd-amdhsa--gfx1100,hipv4-amdgcn-amd-amdhsa--gfx1101,hipv4-amdgcn-amd-amdhsa--gfx1102,hipv4-amdgcn-amd-amdhsa--gfx1151,hipv4-amdgcn-amd-amdhsa--gfx1201 -input=nul "-input=$(IntDir)main_gfx900.o" "-input=$(IntDir)main_gfx906.o" "-input=$(IntDir)main_gfx908.o" "-input=$(IntDir)main_gfx90a.o" "-input=$(IntDir)main_gfx942.o" "-input=$(IntDir)main_gfx1030.o" "-input=$(IntDir)main_gfx1100.o" "-input=$(IntDir)main_gfx1101.o" "-input=$(IntDir)main_gfx1102.o" "-input=$(IntDir)main_gfx1151.o" "-input=$(IntDir)main_gfx1201.o" "-output=$(IntDir)offload_bundle.hipfb"
        cd $(IntDir) && "$(ClangToolPath)llvm-mc" -triple host-x86_64-pc-windows-msvc "hip_obj_gen_win.mcin" -o "main_device.obj" --filetype=obj
    Description: Generating Device Offload Object
    Outputs: $(IntDIr)main_device.obj
    Additional Dependencies: $(IntDir)main_gfx900.o;$(IntDir)main_gfx906.o;$(IntDir)main_gfx908.o;$(IntDir)main_gfx90a.o;$(IntDir)main_gfx942.o;$(IntDir)main_gfx1030.o;$(IntDir)main_gfx1100.o;$(IntDir)main_gfx1101.o;$(IntDir)main_gfx1102.o;$(IntDir)main_gfx1151.o;$(IntDir)main_gfx1201.o;$(IntDir)hip_objgen_win.mcin;%(Inputs)
    Execute Before: ClCompile
    ```

- Finally step 5 is implemented by passing additional inputs to the linker in `project -> properties -> Linker -> Input`:

    ```shell
    Additional Dependencies: $(IntDir)main_device.obj;%(AdditionalDependencies)
    ```

## Dependencies

This example depends on the following tools:

- `clang-offload-bundler`, which is included in the `rocm-llvm` package on Linux.
- `llvm-mc`, which is included in the `rocm-llvm` package on Linux.

`rocm-llvm` is installed with most ROCm installations.

## Used API surface

### HIP runtime

- `hipFree`
- `hipGetDeviceProperties`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
