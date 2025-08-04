# HIP-Doc Simple Device Query Example

## Description

This example shows how the number of HIP-capable devices in the system can be determined, as well as how properties from
the device may be queried. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html#querying-and-setting-gpus).

### Application flow

The number of devices in the system is queried, and the device properties are queried and the device name is printed.

## Key APIs and Concepts

* `hipGetDeviceCount` returns the number of devices in the system. Some device management API functions take an
  identifier for each device, which is a monotonically incrementing number starting from zero. Others require the active
  device to be set, with `hipSetDevice`. A full overview of the device management API can be found at
  [HIP API - Device Management](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___device.html).

## Demonstrated API Calls

### HIP Runtime

* `hipGetDeviceCount`
* `hipGetDeviceProperties`
