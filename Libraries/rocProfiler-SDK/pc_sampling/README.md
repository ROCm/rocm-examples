# rocProfiler SDK PC Sampling

## Description

This example demonstrates how to use the rocProfiler SDK's PC (Program Counter) sampling service. PC sampling periodically interrupts executing GPU wavefronts to record their program counter and other hardware state. This provides a statistical profile of where time is being spent within a kernel, which is invaluable for identifying performance bottlenecks. The tool queries for agents that support PC sampling, configures the service, and processes the resulting sample records.

## Application flow

1. **Tool Loading and Initialization**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool registers its `tool_init` and `tool_fini` functions.
    - The `tool_init` function discovers all GPU agents that support PC sampling and queries their available configurations.
2. **Context, Buffer, and Service Configuration**:
    - A single rocProfiler context is created for the tool.
    - A separate buffer is created for each supported GPU agent.
    - For each agent, the PC sampling service is configured, preferably with the stochastic method.
3. **Context Activation**:
    - The rocProfiler context is started, which activates the PC sampling service on all configured agents.
4. **Workload Execution**:
    - The `main` function in `main.cpp` launches a multi-threaded workload with various kernels to generate GPU activity.
5. **Data Collection and Processing**:
    - As the PC sampling hardware generates data, it is collected into the agent-specific buffers.
    - When a buffer's watermark is reached, the `rocprofiler_pc_sampling_callback` is invoked.
    - The callback processes the records and prints detailed information, including the program counter, workgroup ID, and hardware-specific stall information.
6. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called, which stops the context, flushes and destroys the buffers, and cleans up resources.

## Key APIs and Concepts

- **PC Sampling Service**:
  - `rocprofiler_configure_pc_sampling_service()`: The core of this example. This service configures the hardware to perform PC sampling on a given agent.

- **Sampling Methods**:
  - The tool demonstrates how to query for and select between different sampling methods:
    - `ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`: A low-overhead method where the hardware randomly samples wavefronts.
    - `ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP`: A method that involves the host, which can have higher overhead.

- **Configuration Query**:
  - `rocprofiler_query_pc_sampling_agent_configurations()`: Used to discover the supported sampling methods and their valid interval ranges for a given agent.

- **PC Sample Records**:
  - The callback processes different types of PC sampling records, such as `rocprofiler_pc_sampling_record_stochastic_v0_t`, which contains detailed information about the wavefront state, including whether an instruction was issued and, if not, the reason for the stall.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_assign_callback_thread`
- `rocprofiler_configure_pc_sampling_service`
- `rocprofiler_create_buffer`
- `rocprofiler_create_callback_thread`
- `rocprofiler_create_context`
- `rocprofiler_destroy_buffer`
- `rocprofiler_flush_buffer`
- `rocprofiler_query_available_agents`
- `rocprofiler_query_pc_sampling_agent_configurations`
- `rocprofiler_start_context`

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
- `rocprofiler_pc_sampling_record_header_t`
