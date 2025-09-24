# rocProfiler SDK OpenMP Target Tracing

## Description

This example demonstrates how to use the rocProfiler SDK to trace an application that uses OpenMP target offloading. It captures OMPT events, ROCTX markers, and GPU activities (kernel dispatches, memory copies) to provide a comprehensive view of the application's execution. The tool uses a hybrid approach, with callback tracing for synchronous events like OMPT and ROCTX, and buffered tracing for asynchronous GPU events.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a primary rocProfiler context and a secondary "control" context.
2. **Service Configuration**:
    - The primary context is configured with callback tracing for OMPT events, ROCTX markers, and code object loading.
    - A buffer is created and configured for tracing asynchronous GPU activities (kernel dispatches, memory copies, scratch memory).
    - The control context is configured to handle `roctxProfilerPause` and `roctxProfilerResume` calls.
3. **Context Activation**:
    - Both the primary and control contexts are started.
4. **Workload Execution**:
    - The `main` function in `main.cpp` performs a vector multiplication using OpenMP `#pragma omp target` directives.
    - It also uses ROCTX markers to delineate phases and pauses/resumes profiling.
5. **Event Handling**:
    - The `tool_callback_tracing_callback` is invoked synchronously for OMPT and ROCTX events.
    - The `tool_buffered_tracing_callback` is invoked when the buffer is full, processing the asynchronous kernel and memory copy records.
6. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called, which stops the context, flushes the buffer, and prints the combined trace to a log file.

## Key APIs and Concepts

- **OMPT Tracing**:
  - `ROCPROFILER_CALLBACK_TRACING_OMPT`: This service allows the tool to receive callbacks for OpenMP runtime events, providing insight into how the application is using target offloading.

- **Hybrid Tracing Model**:
  - This example effectively combines two tracing models:
    - **Callback Tracing**: Used for synchronous, host-side events (OMPT, ROCTX) where immediate action or data interaction is needed.
    - **Buffered Tracing**: Used for high-frequency, asynchronous GPU events to minimize overhead.

- **Interacting with OMPT Data**:
  - The tool demonstrates how to access and modify the data pointers provided by the OMPT runtime (e.g., `parallel_data`, `task_data`), allowing for correlation between OMPT events and other trace data.

- **Control Context**:
  - A separate context is used to reliably pause and resume the main tracing context via ROCTX calls.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_buffer_tracing_service`
- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_create_buffer`
- `rocprofiler_create_context`
- `rocprofiler_flush_buffer`
- `rocprofiler_start_context`
- `rocprofiler_stop_context`

### ROCTX

- `roctxRangeStart`
- `roctxRangeStop`
- `roctxMark`
- `roctxGetThreadId`
- `roctxProfilerPause`
- `roctxProfilerResume`

### Data Types and Enums

- `rocprofiler_buffer_id_t`
- `rocprofiler_callback_tracing_record_t`
- `rocprofiler_context_id_t`
- `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
- `ROCPROFILER_CALLBACK_TRACING_OMPT`
- `ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API`
- `ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API`
- `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`
- `ROCPROFILER_BUFFER_TRACING_MEMORY_COPY`
