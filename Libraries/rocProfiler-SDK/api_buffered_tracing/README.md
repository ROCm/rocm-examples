# rocProfiler SDK API and Kernel Buffered Tracing

## Description

This example demonstrates how to use the rocProfiler SDK to trace various activities, including HIP and HSA API calls, kernel dispatches, memory copies, and scratch memory usage. It uses a buffered approach, where trace records are collected in a buffer and processed in batches by a callback function. It also shows how to trace code object loading to map kernel IDs to their names.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - In `main.cpp`, `client::setup()` is called, which forces the rocProfiler runtime to initialize the tool.
    - The `tool_init` function is called, which creates a rocProfiler context, a buffer, and configures the required tracing services (for HIP/HSA APIs, kernel dispatches, etc.).
2. **Starting the Profiler**:
    - `client::start()` is called from `main.cpp` to activate the rocProfiler context before any HIP API calls are made.
3. **Workload Execution**:
    - The `main` function in `main.cpp` creates multiple threads to simulate a workload.
    - Each thread executes three types of operations:
      - `run_migrate`: Demonstrates memory migration using `hipHostRegister`.
      - `run_scratch`: Executes kernels with varying amounts of scratch memory usage.
      - `run_transpose`: Performs a matrix transpose, involving memory allocation, data transfer (`hipMemcpyAsync`), and kernel launches (`hipLaunchKernelGGL`).
    - Throughout the workload, `client::identify()` is called to push an external correlation ID (the thread ID) to associate the work with the trace data.
4. **Data Collection and Processing**:
    - As the application runs, the rocProfiler runtime captures the configured events (API calls, kernel dispatches, etc.) and stores them as records in the buffer.
    - When the buffer's watermark is reached, the `tool_tracing_callback` function is invoked.
    - This callback processes the records, maps kernel IDs to their names (using the data from the `tool_code_object_callback`), and stores the formatted trace information.
5. **Stopping the Profiler and Finalization**:
    - After the workload is complete, `client::stop()` is called from `main.cpp` to deactivate the rocProfiler context.
    - `client::shutdown()` is called, which triggers the `tool_fini` function.
    - `tool_fini` flushes any remaining records from the buffer and writes the collected trace data to a log file.

## Key APIs and Concepts

- **Tool Entry Point**:
  - `rocprofiler_configure()`: This is the primary entry point for a rocProfiler tool. The rocProfiler runtime discovers and calls this function in the tool's shared library. It's used to register the tool's initialization and finalization functions.

- **rocProfiler Initialization**:
  - `rocprofiler_create_context()`: Creates a rocProfiler context, which is a container for tracing and profiling services.
  - `rocprofiler_start_context()` / `rocprofiler_stop_context()`: Activates and deactivates the context.

- **Buffer Tracing**:
  - `rocprofiler_create_buffer()`: Creates a buffer to store trace records.
  - `rocprofiler_configure_buffer_tracing_service()`: Configures a service to collect trace data (e.g., API calls, kernel dispatches) and store it in a buffer.
  - `rocprofiler_flush_buffer()`: Manually flushes the buffer to process any pending records.
  - The `tool_tracing_callback` function is invoked when the buffer's watermark is reached, allowing for batched processing of trace records.

- **Callback Tracing**:
  - `rocprofiler_configure_callback_tracing_service()`: Configures a service that invokes a callback function for specific events. In this example, it's used to trace code object loading.
  - The `tool_code_object_callback` function is used to map kernel IDs to their names, which is essential for interpreting kernel dispatch records.

- **Correlation and Thread Management**:
  - `rocprofiler_push_external_correlation_id()`: Associates a user-defined ID with trace records, which is useful for correlating events in multi-threaded applications.
  - `rocprofiler_create_callback_thread()` and `rocprofiler_assign_callback_thread()`: Manages threads for handling callbacks.
  - `rocprofiler_at_internal_thread_create()`: Registers callbacks for when rocProfiler creates internal threads.

- **Key Enumerations**:
  - `rocprofiler_buffer_tracing_kind_t`: Specifies the type of activity to trace (e.g., `ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API`, `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`).
  - `rocprofiler_callback_tracing_kind_t`: Specifies the type of event for callback tracing (e.g., `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`).
  - `rocprofiler_buffer_policy_t`: Defines the buffer's behavior when it's full (e.g., `ROCPROFILER_BUFFER_POLICY_LOSSLESS`).

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_assign_callback_thread`
- `rocprofiler_at_internal_thread_create`
- `rocprofiler_configure_buffer_tracing_service`
- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_create_buffer`
- `rocprofiler_create_callback_thread`
- `rocprofiler_create_context`
- `rocprofiler_flush_buffer`
- `rocprofiler_get_thread_id`
- `rocprofiler_push_external_correlation_id`
- `rocprofiler_query_buffer_tracing_kind_name`
- `rocprofiler_start_context`
- `rocprofiler_stop_context`

### HIP runtime

- `hipDeviceReset`
- `hipDeviceSynchronize`
- `hipFree`
- `hipGetDeviceCount`
- `hipHostRegister`
- `hipHostUnregister`
- `hipLaunchKernelGGL`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemsetAsync`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `rocprofiler_buffer_id_t`
- `rocprofiler_callback_thread_t`
- `rocprofiler_context_id_t`
- `rocprofiler_tool_configure_result_t`
- `ROCPROFILER_BUFFER_CATEGORY_TRACING`
- `ROCPROFILER_BUFFER_POLICY_LOSSLESS`
- `ROCPROFILER_BUFFER_TRACING_HSA_CORE_API`
- `ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API`
- `ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API`
- `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`
- `ROCPROFILER_BUFFER_TRACING_MEMORY_COPY`
- `ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY`
- `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
