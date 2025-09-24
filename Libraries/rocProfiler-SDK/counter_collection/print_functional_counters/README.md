# rocProfiler SDK Functional Counter Test

## Description

This example serves as a functional test to verify the collection of all available hardware counters on a given GPU agent. It systematically iterates through each supported counter, configures a profiling service to collect it for a single kernel dispatch, and then validates that the correct number of counter instances were received.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context and a buffer.
2. **Workload Execution**:
    - The `main` function in `main.cpp` launches a series of HIP kernels (`kernel_a`, `kernel_b`, `kernel_c`) in a loop. The number of iterations is high to ensure that every available counter is profiled at least once.
3. **Iterative Counter Collection**:
    - The `dispatch_callback` is triggered for each kernel dispatch.
    - On the first dispatch, it queries for all available counters on the agent and stores them in a list.
    - For each subsequent dispatch, it selects the next counter from the list and creates a `rocprofiler_counter_config_id_t` to collect only that counter.
4. **Data Collection and Processing**:
    - The collected counter data is stored in a buffer.
    - When the buffer is full or flushed, the `buffered_callback` is invoked, which records the number of instances seen for each counter ID.
5. **Finalization and Validation**:
    - After the workload is complete, the `tool_fini` function is called.
    - It compares the number of captured counter instances against the expected number (queried from `rocprofiler_query_counter_info`).
    - It reports errors for any counters that are missing or have an incorrect instance count, effectively validating the functionality of each counter.

## Key APIs and Concepts

- **Comprehensive Counter Testing**:
  - The primary purpose is to test the functionality of every hardware counter exposed by the SDK for a given agent.

- **Iterative Profiling**:
  - The `dispatch_callback` acts as a state machine, iterating through a list of available counters and profiling one per dispatch. This allows for systematic testing of all counters.

- **Counter Information Query**:
  - `rocprofiler_query_counter_info()`: Used to get metadata about each counter, including its name and the expected number of instances (`dimensions_instances_count`), which is crucial for validation.
  - `rocprofiler_iterate_agent_supported_counters()`: Used to get the initial list of all available counters for the agent.

- **Validation Logic**:
  - The `tool_fini` function contains detailed logic to check for correctness, comparing expected vs. actual counter data and reporting discrepancies.
  - The `validate_dim_presence` struct is used to ensure that data from all expected hardware instances (e.g., all CUs) is present in the final output.

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
