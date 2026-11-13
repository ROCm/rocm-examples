# HIP-Doc Peer-to-Peer Memory Copy Example

## Description

This example demonstrates how to copy data between two devices with the peer-to-peer copy functions
`hipMemcpyPeer` and `hipMemcpyPeerAsync`. For more information on this topic, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#peer-to-peer-memory-access).

The [peer-to-peer memory access example](../p2p_memory_access) copies between devices with `hipMemcpy` instead.

### Application flow

1. The number of HIP devices is queried and a message printed if there are less than two in the system.
2. Peer-to-peer access is enabled in both directions if the devices support it.
3. A buffer on the source device is filled by a kernel, and the host and the device are synchronized.
4. A buffer is allocated on the destination device.
5. `hipMemcpyPeer` copies the source buffer to the destination device, and the destination device is synchronized.
6. A stream is created on the source device.
7. `hipMemcpyPeerAsync` copies the data back to a second buffer on the source device, and the stream is synchronized.
8. The returned data is copied to the host and validated.
9. The stream and the device memory are freed.

## Key APIs and Concepts

* `hipMemcpyPeer` copies memory between two devices. It takes the device ID of the destination and of the source.
* `hipMemcpyPeerAsync` is the asynchronous version of `hipMemcpyPeer` and runs the copy in the given stream.
* `hipDeviceCanAccessPeer` reports whether a device can directly access memory on another device.
* `hipDeviceEnablePeerAccess` enables peer-to-peer memory access for the given device.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipDeviceCanAccessPeer`
* `hipDeviceEnablePeerAccess`
* `hipDeviceSynchronize`
* `hipFree`
* `hipGetDeviceCount`
* `hipGetErrorString`
* `hipGetLastError`
* `hipMalloc`
* `hipMemcpy`
* `hipMemcpyPeer`
* `hipMemcpyPeerAsync`
* `hipSetDevice`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamSynchronize`
