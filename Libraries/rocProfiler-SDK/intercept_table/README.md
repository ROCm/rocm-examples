# rocProfiler SDK HIP API Intercept Table

## Description

This example demonstrates how to use the rocProfiler SDK's intercept table registration to gain direct control over the HIP runtime API dispatch table. The tool registers a callback that is invoked by the SDK when the HIP runtime is initialized. This callback receives the `HipDispatchTable`, allowing the tool to replace the function pointers for specific HIP API calls with its own wrapper functions.

## Application flow

1. **Tool Loading and Registration**:
    - The rocProfiler runtime loads the tool's shared library and calls the `rocprofiler_configure` entry point.
    - The tool calls `rocprofiler_at_intercept_table_registration` to register the `api_registration_callback` for the HIP runtime API table.
2. **Runtime Initialization and Interception**:
    - When the application initializes the HIP runtime (e.g., by calling `hipGetDeviceCount`), the SDK invokes the `api_registration_callback`.
    - The callback receives a pointer to the `HipDispatchTable`.
    - The tool then overwrites the function pointers for various HIP API calls (e.g., `hipMalloc`, `hipLaunchKernel`) with pointers to its own wrapper functions. The original function pointers are stored to be called from within the wrappers.
3. **Workload Execution**:
    - The `main` function in `main.cpp` runs a multi-threaded matrix transpose workload.
4. **Wrapper Invocation**:
    - Every time the application calls a wrapped HIP API function, the tool's wrapper is executed instead.
    - The wrapper increments a call counter for that specific function.
    - It then calls the original HIP API function using the saved function pointer.
5. **Finalization**:
    - After the workload is complete, the `tool_fini` function is called.
    - It prints a log of all the HIP API calls that were intercepted and the number of times each was called.

## Key APIs and Concepts

- **Intercept Table**:
  - `rocprofiler_at_intercept_table_registration()`: The core of this example. It provides a mechanism to directly modify the dispatch table that the HIP runtime uses to call its own functions.
  - This is used with `ROCPROFILER_HIP_RUNTIME_TABLE` to specify that the HIP runtime API table should be intercepted.

- **Direct API Wrapping**:
  - Unlike callback or buffered tracing, which provide pre- and post-call notifications, intercepting the dispatch table allows a tool to create a true wrapper around the API call.
  - This enables more advanced use cases, such as modifying arguments, filtering calls, or replacing the implementation entirely.

- **`HipDispatchTable`**:
  - This structure, defined in the HIP headers, contains function pointers for all the HIP runtime APIs. The tool modifies this table to insert its wrappers.

## Demonstrated API Calls

### rocProfiler SDK

- `rocprofiler_at_intercept_table_registration`

### HIP runtime

- `hipDeviceReset`
- `hipDeviceSynchronize`
- `hipFree`
- `hipGetDeviceCount`
- `hipGetLastError`
- `hipLaunchKernel`
- `hipMalloc`
- `hipMemcpyAsync`
- `hipMemsetAsync`
- `hipSetDevice`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `rocprofiler_intercept_table_t`
- `ROCPROFILER_HIP_RUNTIME_TABLE`
