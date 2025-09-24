# rocProfiler SDK Code Object ISA Decoding

## Description

This example demonstrates how to use the rocProfiler SDK's code object tracing service to intercept GPU kernel binaries, decode their instruction set architecture (ISA), and perform analysis. The tool captures code object loading events, translates virtual addresses to instructions, and prints a disassembly and instruction statistics for each registered kernel.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context and configures a callback tracing service for code object events.
2. **Context Activation**:
    - The rocProfiler context is started.
3. **Workload Execution**:
    - The `main` function in `main.cpp` launches several variations of a matrix transpose kernel with different data types (`int`, `float`, `double`) and implementations (`transposeNaive`, `transposeLdsSwapInplace`, `transposeLdsNoBankConflicts`).
4. **Code Object Loading and Analysis**:
    - When the HIP runtime loads the kernel binary, the `tool_codeobj_tracing_callback` is invoked with a `ROCPROFILER_CODE_OBJECT_LOAD` operation.
    - The tool uses the `CodeobjAddressTranslate` utility from the SDK to create a decoder for the loaded code object.
    - Subsequently, the callback is invoked with a `ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER` operation for each kernel.
    - The tool then iterates through the kernel's memory range, decodes the ISA for each virtual address, prints a partial disassembly, and collects instruction statistics (e.g., counts of scalar, vector, and wait instructions).
5. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called, and the analysis results are written to a log file.

## Key APIs and Concepts

- **Code Object Tracing**:
  - `rocprofiler_configure_callback_tracing_service()`: Used with the `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT` kind to register a callback for code object lifecycle events.
  - `ROCPROFILER_CODE_OBJECT_LOAD`: The operation kind that indicates a code object has been loaded into memory.
  - `ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`: The operation kind that provides the mapping between a kernel name and its implementation within the code object.

- **ISA Decoding**:
  - `rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate`: A C++ utility provided by the SDK that is essential for decoding the ISA. It builds a mapping from virtual addresses to human-readable instructions and their properties.
  - `rocprofiler::sdk::codeobj::disassembly::Instruction`: A data structure representing a single decoded instruction, including its text, size, and comments.

- **Instruction-Level Analysis**:
  - By decoding the ISA, the tool can perform low-level analysis, such as counting instruction types (scalar, vector, wait), which can be valuable for performance tuning and understanding kernel behavior.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_create_context`
- `rocprofiler_start_context`

### HIP runtime

- `hipDeviceReset`
- `hipDeviceSynchronize`
- `hipEventCreate`
- `hipEventDestroy`
- `hipEventElapsedTime`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipFree`
- `hipGetDeviceCount`
- `hipGetLastError`
- `hipLaunchKernelGGL`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemsetAsync`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `rocprofiler_callback_tracing_record_t`
- `rocprofiler_context_id_t`
- `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
- `ROCPROFILER_CODE_OBJECT_LOAD`
- `ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`
