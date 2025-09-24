# rocProfiler SDK Code Object Tracing

## Description

This example demonstrates how to use the rocProfiler SDK's callback tracing service to monitor the lifecycle of GPU code objects. The tool sets up a callback to receive notifications whenever a code object is loaded or unloaded, and whenever a kernel symbol is registered or unregistered within that code object.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context and configures a callback tracing service for code object events.
2. **Context Activation**:
    - The rocProfiler context is started.
3. **Workload Execution**:
    - The `main` function in `main.cpp` launches a multi-threaded matrix transpose workload. This action triggers the loading of the GPU kernel binary (code object) by the HIP runtime.
4. **Event Handling**:
    - The `tool_tracing_callback` is invoked for each event in the code object's lifecycle:
      - **`ROCPROFILER_CODE_OBJECT_LOAD`**: When the binary is loaded, the callback logs details such as the code object ID, URI, and memory layout.
      - **`ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`**: For each kernel in the binary, the callback logs its properties, including the demangled kernel name, segment sizes, and object IDs.
5. **Finalization**:
    - After the workload is complete and the application exits, the `tool_fini` function is called.
    - This function prints the entire captured trace of code object events to a log file.

## Key APIs and Concepts

- **Code Object Tracing**:
  - `rocprofiler_configure_callback_tracing_service()`: Used with the `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT` kind to register a callback for code object lifecycle events.
  - This service provides a direct way to observe the runtime loading and unloading of GPU binaries.

- **Lifecycle Events**:
  - The tool handles both `ROCPROFILER_CALLBACK_PHASE_LOAD` and `ROCPROFILER_CALLBACK_PHASE_UNLOAD` phases for both code objects and kernel symbols, allowing for a complete trace of their lifecycle.
  - `rocprofiler_callback_tracing_code_object_load_data_t`: Provides information about the loaded code object.
  - `rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t`: Provides detailed information about each kernel, such as its memory requirements (`kernarg_segment_size`, `group_segment_size`, `private_segment_size`).

- **C++ Demangling**:
  - The example uses `abi::__cxa_demangle` to convert the mangled C++ kernel names provided by the runtime into a human-readable format, which is useful for logging and analysis.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_create_context`
- `rocprofiler_start_context`

### HIP runtime

- `hipDeviceReset`
- `hipDeviceSynchronize`
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
