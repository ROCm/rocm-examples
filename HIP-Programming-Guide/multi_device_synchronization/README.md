# HIP-Programming-Guide Multi-Device Synchronization Example

## Description

In a multi-device system, streams and events are essential for efficient parallel computation and synchronization.
Streams enable asynchronous task execution, allowing multiple devices to process data concurrently without blocking one
another. Events provide a mechanism for synchronizing operations across streams and devices, ensuring that tasks on one
device are completed before dependent tasks on another device begin. This coordination prevents race conditions and
optimizes data flow in multi-GPU systems. Together, streams and events maximize performance by enabling parallel
execution, load balancing, and effective resource utilization across heterogeneous hardware.

This example demonstrates how to synchronize multiple devices using HIP events and streams.

### Application flow

1. The number of HIP devices is queried and an error printed if there are less than two in the system.
2. Two devices are initialized sequentially:
    1. The device is set as active device.
    2. A HIP stream is created.
    3. Two HIP events are created.
    4. Device memory is allocated.
3. Tasks are performed sequentially on both devices:
    1. The device is set as active device.
    2. The first event is recorded in the stream.
    3. A kernel is launched on the device.
    4. The second event is recorded in the stream.
    5. The host is synchronized by waiting for the second event.
4. For each device the elapsed time between the first and second event is queried and printed.
5. Both devices are cleaned up sequentially:
    1. The device is set as active device.
    2. Both events are destroyed.
    3. The stream is synchronized and then destroyed.
    4. The device memory is freed.

## Key APIs and Concepts

* `hipSetDevice` sets the active device. All succeeding HIP API calls will target this device until `hipSetDevice` is
  called again or the application exits.
* `hipEventRecord` records an event in a HIP stream. Events can be used for synchronization with the host, a stream from
  another device, or a stream from the same device.
* `hipEventSynchronize` synchronizes the host by blocking until the given event is reached by the stream.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipFree`
* `hipGetErrorString`
* `hipMalloc`
* `hipMemcpy`
* `hipSetDevice`
