# rocProfiler SDK API Callback Tracing

## Description

This example demonstrates how to trace HIP, HSA, and ROCTX API calls using a direct callback-based approach. For each API call, a callback is invoked upon entry and exit, allowing for immediate, synchronous processing of the event. This sample also illustrates how to use a separate "control" context to pause and resume tracing based on ROCTX API calls.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function is called, which creates a primary rocProfiler context (`client_ctx`) and a secondary "control" context.
2. **Context Configuration**:
    - The primary context is configured with a callback (`tool_tracing_callback`) to trace various API services, including HSA, HIP, and ROCTX.
    - The control context is configured with a separate callback (`tool_tracing_ctrl_callback`) that only listens for `roctxProfilerPause` and `roctxProfilerResume` API calls.
3. **Context Activation**:
    - Both the primary and control contexts are started.
4. **Workload Execution**:
    - The `main` function in `main.cpp` creates multiple threads to run a matrix transpose workload.
    - The workload is wrapped in ROCTX ranges (`roctxRangeStart`/`roctxRangeStop`) for visualization in profiling tools.
    - The application calls various HIP APIs for memory management (`hipMalloc`, `hipMemcpyAsync`) and kernel execution (`hipLaunchKernelGGL`).
5. **Tracing and Control**:
    - As the application runs, the `tool_tracing_callback` is invoked for each entry and exit of the traced API calls, logging the event details.
    - Before shutting down, the application calls `roctxProfilerPause`, which triggers the `tool_tracing_ctrl_callback` to stop the primary context.
    - A call to `hipDeviceReset` is made, which is *not* traced because the primary context is paused.
    - `roctxProfilerResume` is then called, which triggers the control callback to restart the primary context.
6. **Finalization**:
    - `client::shutdown()` is called, which triggers the `tool_fini` function.
    - `tool_fini` stops the primary context and writes the collected trace data to a log file.

## Key APIs and Concepts

- **Callback Tracing**:
  - `rocprofiler_configure_callback_tracing_service()`: The core of this example. It registers a callback function that is invoked synchronously for each entry and exit of the specified API calls.
  - `rocprofiler_callback_tracing_record_t`: The data structure passed to the callback, containing information about the event, such as the API kind, operation, thread ID, and phase (enter/exit).

- **Dual-Context Control**:
  - A key feature is the use of two contexts: a primary context for general tracing and a "control" context dedicated to pausing and resuming the primary context.
  - This avoids a deadlock where pausing a context would also disable the callback needed to resume it.

- **Argument Introspection**:
  - `rocprofiler_iterate_callback_tracing_kind_operation_args()`: Allows for inspecting the arguments of the traced API calls within the callback, enabling deep analysis of the application's behavior.

- **ROCTX Integration**:
  - The example traces ROCTX APIs to demonstrate how to correlate application-level events with low-level API and GPU activities.
  - `roctxProfilerPause` and `roctxProfilerResume` are used to dynamically control the tracing process.

- **Key Enumerations**:
  - `rocprofiler_callback_tracing_kind_t`: Specifies the type of API to trace (e.g., `ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API`, `ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API`).
  - `rocprofiler_callback_phase_t`: Indicates whether the callback is for the entry (`ROCPROFILER_CALLBACK_PHASE_ENTER`) or exit (`ROCPROFILER_CALLBACK_PHASE_EXIT`) of an API call.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_create_context`
- `rocprofiler_iterate_callback_tracing_kind_operation_args`
- `rocprofiler_start_context`
- `rocprofiler_stop_context`

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

### ROCTX

- `roctxRangeStart`
- `roctxRangeStop`
- `roctxRangePush`
- `roctxRangePop`
- `roctxMark`
- `roctxGetThreadId`
- `roctxProfilerPause`
- `roctxProfilerResume`

### Data Types and Enums

- `rocprofiler_callback_tracing_record_t`
- `rocprofiler_context_id_t`
- `ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API`
- `ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API`
- `ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API`
- `ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API`
- `ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API`
