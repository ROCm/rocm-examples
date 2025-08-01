# HIP-Documentation Per-Thread Default Stream Example

## Description

This example demonstrates how to manage streams on a per-thread basis. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_driver_api.html#per-thread-default-stream-version-request).

### Application flow

1. The HIP runtime is initialized.
2. The default stream for the current thread is obtained.
3. A device buffer is allocated.
4. An asynchronous `memset` operation is placed in the stream.
5. The stream is synchronized with the host.
6. The device buffer is freed.

## Key APIs and Concepts

* `hipStreamPerThread` is a host symbol which always points to the default stream for the calling thread.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipFree`
* `hipInit`
* `hipMalloc`
* `hipMemsetAsync`
* `hipStreamPerThread`
