# rocProfiler SDK Device Counter Collection

## Description

This example demonstrates how to use the rocProfiler SDK for asynchronous, device-level hardware counter collection. Unlike dispatch-specific profiling, this method samples counters periodically on a separate thread. The application configures a device counting service, which runs in the background and collects data from a specified GPU agent.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context, a buffer for the counter data, and discovers the first available GPU agent.
2. **Profile and Service Configuration**:
    - A counter profile for `SQ_WAVES` is built and cached for the selected agent.
    - The `rocprofiler_configure_device_counting_service` is configured. It uses a `set_profile` callback to provide the counter configuration to the service.
3. **Asynchronous Sampling**:
    - A detached thread is launched, which starts the rocProfiler context.
    - Inside this thread, `rocprofiler_sample_device_counting_service` is called in a loop to trigger counter collection periodically (every 50ms).
4. **Workload Execution**:
    - Concurrently, the `main` function in `main.cpp` launches a series of HIP kernels (`kernel_a`, `kernel_b`, `kernel_c`) in a loop to generate GPU activity.
5. **Data Collection and Processing**:
    - The sampling thread collects the counter data and stores it in the buffer.
    - When the buffer's watermark is reached, the `buffered_callback` is invoked to process and print the counter values.
6. **Finalization**:
    - After the workload is complete, the `tool_fini` function signals the sampling thread to stop, stops the context, and flushes the buffer to ensure all data is processed.

## Key APIs and Concepts

- **Device Counting Service**:
  - `rocprofiler_configure_device_counting_service()`: Configures a service for profiling an entire GPU device asynchronously.
  - The service is configured for a single `rocprofiler_agent_id_t`, allowing for targeted profiling of one GPU.

- **Asynchronous Sampling**:
  - `rocprofiler_sample_device_counting_service()`: Explicitly triggers a counter collection sample. In this example, it is called from a dedicated thread to perform periodic, time-based monitoring, independent of kernel dispatches.

- **Profile Callback**:
  - A `set_profile` callback is used by the service to request the `rocprofiler_counter_config_id_t` for the targeted agent.

- **Buffer and Callback**:
  - A `rocprofiler_buffer_id_t` is used to store the sampled counter data, which is then processed by a `buffered_callback`.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_assign_callback_thread`
- `rocprofiler_configure_device_counting_service`
- `rocprofiler_create_buffer`
- `rocprofiler_create_callback_thread`
- `rocprofiler_create_context`
- `rocprofiler_create_counter_config`
- `rocprofiler_flush_buffer`
- `rocprofiler_iterate_agent_supported_counters`
- `rocprofiler_query_available_agents`
- `rocprofiler_query_counter_info`
- `rocprofiler_sample_device_counting_service`
- `rocprofiler_start_context`
- `rocprofiler_stop_context`

### HIP runtime

- `hipDeviceReset`
- `hipDeviceSynchronize`
- `hipFree`
- `hipGetDeviceCount`
- `hipGetDeviceProperties`
- `hipLaunchKernelGGL`
- `hipMalloc`
- `hipMemcpy`
- `hipSetDevice`

### Data Types and Enums

- `rocprofiler_buffer_id_t`
- `rocprofiler_callback_thread_t`
- `rocprofiler_context_id_t`
- `rocprofiler_counter_config_id_t`
- `ROCPROFILER_BUFFER_CATEGORY_COUNTERS`
- `ROCPROFILER_BUFFER_POLICY_LOSSLESS`
- `ROCPROFILER_COUNTER_RECORD_VALUE`
