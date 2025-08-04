# HIP-Doc Device Enumeration Example

## Description

This example demonstrates how to query the number of devices in the system and how to access them. For more information
on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#device-enumeration).

### Application flow

1. The number of HIP devices is queried.
2. Loop over the number of devices:
    1. Query the current device's properties.
    2. Print the properties.

## Key APIs and Concepts

* `hipGetDeviceCount` obtains the number of HIP devices in the system.
* `hipGetDeviceProperties` returns the properties of a given device.

## Demonstrated API Calls

### HIP Runtime

#### Host symbols

* `hipGetDeviceCount`
* `hipGetDeviceProperties`
