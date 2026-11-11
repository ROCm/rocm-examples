# rocProfiler SDK Synchronous Device Counter Collection

## Description

This example demonstrates synchronous, on-demand, device-level hardware counter collection using the rocProfiler SDK. It features a `counter_sampler` class that encapsulates the complexity of setting up profiling, creating counter configurations, and sampling data. The application uses this class to periodically sample counters from a separate thread and print the results.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function discovers the first available GPU agent and initializes a `counter_sampler` object for it.
2. **Sampler Initialization**:
    - The `counter_sampler` constructor creates a rocProfiler context, a buffer, and configures the device counting service.
    - Before starting the sampling thread, `tool_init` calls `create_profile` for the requested counters (`SQ_WAVES`) to create and cache the profile. This completes the SDK's one-time counter setup before the workload runs.
    - It then registers an `atexit` handler, so the SDK static objects initialized during profile creation remain alive until the handler finishes.
3. **Asynchronous Sampling**:
    - A joinable thread is launched to perform the counter sampling.
    - Inside this thread, the `sample_counter_values` method is called in a loop. This method is synchronous:
        - It calls `create_profile`, which reuses the cached profile. If any requested counter is unavailable, it clears the output vector and returns `ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS` without sampling a partial profile.
        - It starts the rocProfiler context.
        - It calls `rocprofiler_sample_device_counting_service` to collect the data, which blocks until the data is ready.
        - It stops the rocProfiler context.
        - It returns the collected data in an output vector.
4. **Workload Execution**:
    - Concurrently, the `main` function in `main.cpp` launches a series of HIP kernels (`kernel_a`, `kernel_b`, `kernel_c`) in a loop to generate GPU activity.
5. **Data Processing**:
    - The sampling thread receives the `rocprofiler_counter_record_t` vector and prints the counter names, values, and dimensions.
6. **Finalization**:
    - After the workload is complete, the `atexit` handler signals the sampling thread to exit and joins it before calling the SDK finalization function. This prevents the thread from using SDK services or counter metadata during teardown.
    - The `tool_fini` function also stops and joins the thread if it is still joinable, then stops the context, flushes the buffer and output stream, and releases the sampler and thread object.

## Key APIs and Concepts

- **Synchronous Sampling**:
  - `rocprofiler_sample_device_counting_service()`: The core of this example. The `out.data()` parameter is used to receive the counter data directly, and the function blocks until the sampling is complete.

- **`counter_sampler` Class**:
  - This class provides a high-level abstraction for device profiling. It manages the rocProfiler context, buffer, and profile configurations internally.

- **Profile Creation and Reuse**:
  - The `create_profile` method creates and caches `rocprofiler_counter_config_id_t` profiles by the requested counter names, returning `ROCPROFILER_STATUS_SUCCESS` for a new or cached profile. It rejects the entire request if any counter is unavailable.
  - This example creates its profile in `tool_init`, before registering the exit handler and launching the sampling thread. `sample_counter_values` reuses that profile and propagates profile creation failures to its caller.

- **Context Lifecycle per Sample**:
  - The rocProfiler context is started and stopped around each call to `rocprofiler_sample_device_counting_service`, ensuring that counters are only collected when explicitly requested.

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
- `rocprofiler_query_record_counter_id`
- `rocprofiler_query_record_dimension_position`
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
- `ROCPROFILER_BUFFER_POLICY_LOSSLESS`
