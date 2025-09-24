# rocProfiler SDK Thread Trace

## Description

This example demonstrates how to use the rocProfiler SDK's experimental device thread trace service. Thread tracing provides a detailed, instruction-by-instruction log of a wavefront's execution, including program counter (PC) and latency information. This sample configures the thread trace service, uses a separate decoder library to parse the raw trace data, and generates a report of the top performance hotspots based on instruction latency.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates two rocProfiler contexts: one for tracing and one for the thread trace service. It also initializes the thread trace decoder.
2. **Service Configuration**:
    - A callback tracing service is configured on the tracing context to intercept code object loading events, which are needed by the decoder.
    - Another callback tracing service is configured on the tracing context to handle `roctxProfilerPause` and `roctxProfilerResume` calls, which start and stop the thread trace context.
    - For each available GPU agent, the thread trace service is configured on the agent context.
3. **Context Activation**:
    - The tracing context is started. The thread trace context is started later by a `roctxProfilerResume` call.
4. **Workload Execution**:
    - The `main` function in `main.cpp` runs several different kernels. It uses `roctxProfilerResume` to start the thread trace and `roctxProfilerPause` to stop it.
5. **Data Collection and Decoding**:
    - The `shader_data_callback` receives the raw trace data from the hardware and passes it to the decoder.
    - The decoder invokes a `parse` lambda for different record types, which aggregates the instruction latencies and hit counts for each program counter.
6. **Finalization**:
    - After the workload is complete, the `tool_fini` function analyzes the collected data, sorts the instructions by total latency, and prints a report of the top 50 hotspots to a log file.

## Key APIs and Concepts

- **Thread Trace Service**:
  - `rocprofiler_configure_device_thread_trace_service()`: The core of this example. This experimental service provides the most detailed level of execution tracing available.

- **Trace Decoder**:
  - `rocprofiler_thread_trace_decoder_create()` and `rocprofiler_trace_decode()`: Used to load and interact with a separate decoder library that can parse the raw, compressed trace data into a structured format.

- **Instruction-Level Latency**:
  - The decoded trace provides per-instruction latency information, allowing for precise identification of performance bottlenecks.

- **ROCTX Control**:
  - The start and stop of the thread trace is controlled via ROCTX calls (`roctxProfilerResume`/`roctxProfilerPause`), demonstrating how to precisely target the code region to be profiled.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_configure_device_thread_trace_service`
- `rocprofiler_create_context`
- `rocprofiler_query_available_agents`
- `rocprofiler_start_context`
- `rocprofiler_stop_context`
- `rocprofiler_thread_trace_decoder_create`
- `rocprofiler_thread_trace_decoder_destroy`
- `rocprofiler_trace_decode`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipLaunchKernelGGL`
- `hipMalloc`
- `hipMemset`
- `hipStreamCreateWithFlags`
- `hipStreamDestroy`

### ROCTX

- `roctxProfilerPause`
- `roctxProfilerResume`

### Data Types and Enums

- `rocprofiler_callback_tracing_record_t`
- `rocprofiler_context_id_t`
- `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
- `ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API`
