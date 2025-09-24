# rocProfiler SDK Callback Counter Collection

## Description

This example demonstrates how to use the rocProfiler SDK to collect hardware counters from kernel dispatches using a direct callback-based approach. It configures a dispatch callback that determines which counters to collect for each kernel and a record callback that receives the counter data directly upon completion of the dispatch.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function creates a rocProfiler context for the profiling session.
2. **Service Configuration**:
    - The `rocprofiler_configure_callback_dispatch_counting_service` is configured. This service uses two key callbacks:
      - `dispatch_callback`: Invoked when a kernel is dispatched. It is responsible for creating a counter profile (in this case, for `SQ_WAVES`) and returning a configuration ID.
      - `record_callback`: Invoked after the kernel has finished executing. It receives the collected counter data.
3. **Context Activation**:
    - The profiling context is started via `rocprofiler_start_context`, enabling the callbacks.
4. **Workload Execution**:
    - The `main` function in `main.cpp` launches a series of HIP kernels (`kernel_a`, `kernel_b`, `kernel_c`) in a loop to generate GPU activity.
5. **Data Collection and Processing**:
    - For each kernel dispatch, the `dispatch_callback` is invoked, which provides the counter configuration.
    - After the kernel finishes execution, the `record_callback` receives the counter data as an array of `rocprofiler_counter_record_t` and prints it to the console.
6. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called, which stops the profiling context.

## Key APIs and Concepts

- **Callback-Based Counter Collection**:
  - `rocprofiler_configure_callback_dispatch_counting_service()`: The core of this example. It provides a direct mechanism for collecting counters without an intermediate buffer, using a dispatch callback and a record callback.

- **Dispatch Callback**:
  - The `dispatch_callback` function is responsible for dynamically deciding which counters to collect for a given kernel dispatch. It creates a `rocprofiler_counter_config_id_t` on the fly if one is not already cached for the agent.

- **Record Callback**:
  - The `record_callback` function receives the counter data directly after the dispatch is complete, making it suitable for immediate, synchronous processing.

- **Counter Configuration**:
  - `rocprofiler_create_counter_config()`: Used within the dispatch callback to define the set of counters to be collected.
  - `rocprofiler_iterate_agent_supported_counters()`: Used to discover the available hardware counters on a given GPU agent.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_configure_callback_dispatch_counting_service`
- `rocprofiler_create_context`
- `rocprofiler_create_counter_config`
- `rocprofiler_iterate_agent_supported_counters`
- `rocprofiler_query_counter_info`
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

- `rocprofiler_context_id_t`
- `rocprofiler_counter_config_id_t`
- `rocprofiler_counter_record_t`
