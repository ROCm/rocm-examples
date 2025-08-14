# HIP-Doc Call Stack Management Example

## Description

This example demonstrates how to adjust the device's call stack size. For more information on this topic, please refer
to the [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/call_stack.html#call-stack).

### Application flow

1. The current stack size limit is queried and printed.
2. A new stack size limit is set.
3. The updated stack size limit is queried and printed.

## Key APIs and concepts

* `hipDeviceGetLimit` queries the device for a requested limit, in this case the stack size.
* `hipDeviceSetLimit` sets a new limit for the device.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipDeviceGetLimit`
* `hipDeviceSetLimit`
* `hipGetErrorString`
