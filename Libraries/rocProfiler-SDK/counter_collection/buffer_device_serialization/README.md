# rocProfiler SDK Buffer Counter Collection with Device Serialization

## Description

This example demonstrates how to use the rocProfiler SDK to collect hardware counters from kernel dispatches across multiple devices, ensuring that the dispatches are serialized on a per-device basis. This is a more advanced example that builds on the concepts of buffer-based counter collection.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context, a buffer for storing counter data, and a dedicated thread for buffer callbacks.
2. **Agent and Counter Discovery**:
    - The tool queries for available GPU agents on the system.
    - For each GPU agent, it iterates through the supported hardware counters and creates a counter profile to collect the `TCC_HIT` counter.
3. **Service Configuration**:
    - The `rocprofiler_configure_buffer_dispatch_counting_service` is configured. This service uses a `dispatch_callback` that is invoked for each kernel dispatch to select the appropriate counter profile.
4. **Context Activation**:
    - The profiling context is started, enabling the interception of kernel dispatches.
5. **Workload Execution**:
    - The `main` function in `main.cpp` launches a kernel on two different devices. The kernels are designed to wait on a shared value, demonstrating the need for per-device serialization.
6. **Data Collection and Processing**:
    - As the kernels execute, the rocProfiler runtime collects the `TCC_HIT` counter data and stores it in the buffer.
    - When the buffer's watermark is reached, the `buffered_callback` function is invoked on the dedicated callback thread.
    - This callback processes the records, filters for counter data, and prints the results.
7. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called.
    - It flushes the buffer to process any remaining records and stops the profiling context.

## Key APIs and Concepts

- **Buffer-Based Counter Collection**:
  - `rocprofiler_configure_buffer_dispatch_counting_service()`: Configures the service for collecting counter data into a buffer.
  - `rocprofiler_create_buffer()`: Creates a buffer to store the counter records.
  - A `buffered_callback` function is registered to process the records when the buffer is full or flushed.

- **Dispatch Callback**:
  - A `dispatch_callback` function is provided to the service. It is invoked for each kernel dispatch and is responsible for returning a `rocprofiler_counter_config_id_t`, which specifies which counters to collect for that dispatch.

- **Counter Configuration**:
  - `rocprofiler_create_counter_config()`: Creates a profile of counters to be collected for a specific agent.
  - `rocprofiler_iterate_agent_supported_counters()`: Used to discover the available hardware counters on a given GPU agent.

- **Multi-Device Serialization**:
  - This example implicitly demonstrates the importance of per-device serialization when profiling multi-GPU applications. Although no specific API is called for this, the correct functioning of the example relies on the rocProfiler runtime's ability to handle dispatches to multiple devices in a serialized manner.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_assign_callback_thread`
- `rocprofiler_configure_buffer_dispatch_counting_service`
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
- `rocprofiler_start_context`
- `rocprofiler_stop_context`

### HIP runtime

- `hipDeviceSynchronize`
- `hipGetDeviceCount`
- `hipLaunchKernelGGL`
- `hipMallocManaged`
- `hipSetDevice`

### Data Types and Enums

- `rocprofiler_buffer_id_t`
- `rocprofiler_callback_thread_t`
- `rocprofiler_context_id_t`
- `rocprofiler_counter_config_id_t`
- `ROCPROFILER_BUFFER_CATEGORY_COUNTERS`
- `ROCPROFILER_BUFFER_POLICY_LOSSLESS`
- `ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER`
- `ROCPROFILER_COUNTER_RECORD_VALUE`
