# HIP-Doc Device Enumeration Example

## Description

Device enumeration involves identifying all the available GPUs connected to the host system. A single host machine can
have multiple GPUs, each with its own unique identifier. By listing these devices, you can decide which GPU to use for
computation. The host queries the system to count and list all connected GPUs that support the chosen `HIP_PLATFORM`,
ensuring that the application can leverage the full computational power available. Typically, applications list devices
and their properties for deployment planning, and also make dynamic selections during runtime to ensure optimal
performance.

If the application does not define a specific GPU, device 0 is selected by default.

This example demonstrates how to query the number of devices in the system and how to access them.

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
