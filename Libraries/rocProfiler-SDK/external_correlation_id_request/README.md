# rocProfiler SDK External Correlation ID Request

## Description

This example demonstrates how to use the rocProfiler SDK's external correlation ID service. This feature allows a tool to be notified when an asynchronous GPU operation (like a kernel dispatch or memory copy) is about to be launched. The tool can then generate and associate a custom "external" correlation ID with that operation, which is later delivered with the corresponding record in the trace buffer. This provides a robust mechanism for linking GPU activity back to the specific CPU context that initiated it.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context, a buffer, and configures the necessary tracing and correlation ID services.
2. **Service Configuration**:
    - The `rocprofiler_configure_external_correlation_id_request_service` is configured for kernel dispatches and memory copies, with `set_external_correlation_id` as the callback.
    - Buffer tracing is enabled for HIP APIs, kernel dispatches, memory copies, and correlation ID retirement events.
3. **Context Activation**:
    - The context is started.
4. **Workload Execution**:
    - The `main` function in `main.cpp` launches a multi-threaded workload that performs memory migrations, uses scratch memory, and executes a matrix transpose kernel.
5. **Correlation ID Request and Tracing**:
    - Before a kernel dispatch or async memory copy is enqueued, the SDK invokes the `set_external_correlation_id` callback.
    - The callback allocates a custom data structure and sets it as the external correlation ID.
    - When the asynchronous operation completes, the `tool_tracing_callback` receives the trace record, which now contains the custom external correlation ID.
6. **Finalization and Validation**:
    - After the workload is complete, the `tool_fini` function is called.
    - It validates that every external correlation ID that was requested was seen in a buffer record, and that every internal correlation ID was eventually retired.

## Key APIs and Concepts

- **External Correlation ID Service**:
  - `rocprofiler_configure_external_correlation_id_request_service()`: The core of this example. It provides a pre-dispatch callback that allows the tool to inject its own correlation data.
  - This service is the primary mechanism for reliably linking asynchronous GPU operations back to the CPU thread and context that initiated them.

- **Correlation ID Lifecycle**:
  - The example demonstrates the full lifecycle of a correlation ID:
    1. The SDK creates an internal ID.
    2. The request service provides this internal ID to the tool.
    3. The tool creates an external ID.
    4. The buffer record contains both.
    5. A `ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT` record signals that the internal ID is no longer in use.

- **Data Validation**:
  - The `tool_fini` function performs rigorous checks to ensure that the correlation ID mechanism is working correctly, which is a good practice for tool developers.
  - The temporal-ordering check between operation end timestamps and correlation ID retirement timestamps is reported but non-fatal by default, because GPU/CPU clock-domain skew on some hardware (for example RDNA3.5 APUs) produces benign timestamp inversions that rocprofiler itself only warns about. Set the environment variable `ROCPROFILER_CI_STRICT_TIMESTAMPS=1` to turn a violation into a hard failure.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_buffer_tracing_service`
- `rocprofiler_configure_callback_tracing_service`
- `rocprofiler_configure_external_correlation_id_request_service`
- `rocprofiler_create_buffer`
- `rocprofiler_create_context`
- `rocprofiler_flush_buffer`
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
- `rocprofiler_context_id_t`
- `ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API`
- `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`
- `ROCPROFILER_BUFFER_TRACING_MEMORY_COPY`
- `ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT`
- `ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH`
- `ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY`
