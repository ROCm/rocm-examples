# HIP-Basic Multi GPU Data Transfer Example

## Description

Peer-to-peer (P2P) communication allows direct communication over PCIe (or NVLINK, in some NVIDIA configurations) between devices. Given that it is not necessary to access the host in order to transfer data between devices, P2P communications provide a lower latency than traditional communications that do need to access the host.

Because P2P communication is done over PCIe/NVLINK, the availability of this type of communication among devices depends mostly on the PCIe/NVLINK topology existing.

In this example, the result of a matrix transpose kernel execution on one device is directly copied to the other one, showcasing how to carry out a P2P communication between two GPUs.

### Application flow

1. P2P communication support is checked among the available devices. In case two of these devices are found to have it between them, they are selected for the example. A trace message informs about the IDs of the devices selected.
2. The input and output matrices are allocated and initialized in host memory.
3. The first device selected is set as the current device, device memory for the input and output matrices is allocated on the current device and the input data is copied from the host.
4. A matrix transpose kernel using static shared memory is then launched on the current device using the previously defined arguments. A synchronization function is used to wait for the kernel to finish before continuing the host execution.
5. The second device selected is set as the current device and the necessary amount of device memory for the input and output matrices for the second kernel execution is allocated on the current device.
6. Direct memory access is enabled from the second device to the first one. This allows memory to be copied from the first device to the second with the usual memory-copy functions.
7. The input data is copied to the current device from the output matrix allocated on the first device and a matrix transpose kernel using dynamic shared memory is then launched on the current device. A synchronization function is used to wait for the kernel to finish before continuing the host execution.
8. The output matrix from this second kernel execution is copied back to host memory.
9. Direct memory access from the second device to the first one is disabled.
10. Device memory is freed.
11. Results are validated and printed to the standard output.

## Key APIs and Concepts

- `hipGetDeviceCount` gives the number of devices available. In this example it allows to check if there is more than one device available.
- `hipDeviceCanAccessPeer` queries whether a certain device can directly access the memory of a given peer device. A P2P communication is supported between two devices if this function returns true for those two devices.
- `hipSetDevice` sets the specified device as the default device for the subsequent API calls. Such a device is then known as _current device_.
- Once a current device is selected, and if a P2P communication is possible with a certain peer device, `hipDeviceEnablePeerAccess` can be used to enable access from the current device to the peer device's memory. With `hipDeviceDisablePeerAccess` it can also be disabled (provided that `hipDeviceEnablePeerAccess` has already been called for the same current and peer devices).
- `hipMalloc` allocates memory in the global memory of the device. When `hipSetDevice` is called to set a specific current device, the subsequent calls to `hipMalloc` will allocate memory in the current device's global memory. In the example it is showcased how to use these two functions to allocate memory on the two devices used.
- With `hipMemcpy` data bytes can be transferred from host to device (using `hipMemcpyHostToDevice`), from device to host (using `hipMemcpyDeviceToHost`) or from device to device (using `hipMemcpyDeviceToDevice`). The latter will only work if P2P communication has been enabled from the destination to the source device.
- `myKernelName<<<...>>>` queues the execution of a kernel in the current device and `hipDeviceSynchronize` makes the host to wait on all active streams on the current device. In this example `hipDeviceSynchronize` is necessary because the second device needs the results obtained from the previous kernel execution on the first device.
- `hipDeviceReset` discards the state of the current device and updates it to fresh one. It also frees all the resources (e.g. streams, events, ...) associated with the current device.
- It's a [known issue with multi-GPU environments](https://community.amd.com/t5/knowledge-base/iommu-advisory-for-amd-instinct/ta-p/484601) that some multi-GPU environments fail due to limitations of the IOMMU enablement, so it may be needed to explicitly enable/disable the IOMMU using the kernel command-line parameter `iommu=pt/off`.

## Demonstrated API Calls

### HIP runtime

- `__global__`
- `__shared__`

#### Device symbols

- `blockDim`
- `blockIdx`
- `threadIdx`
- `__syncthreads`

#### Host symbols

- `hipDeviceCanAccessPeer`
- `hipDeviceDisablePeerAccess`
- `hipDeviceEnablePeerAccess`
- `hipDeviceReset`
- `hipDeviceSynchronize`
- `hipFree`
- `hipGetDeviceCount`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToDevice`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipSetDevice`
- `hipStreamDefault`
- `hipSuccess`
