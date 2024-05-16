# HIP-Basic Device Globals Example

## Description

This program showcases a simple example that uses device global variables to perform a simple test kernel. Two such global variables are set using different methods: one is a single variable is set by first obtaining a pointer to it and using `hipMemcpy`, as would be done for a pointer to device memory using `hipMalloc`. The other is an array that is initialized without first explicitly obtaining the pointer by using `hipMemcpyToSymbol`.

### Application flow

1. A number of constants are defined for the kernel launch parameters.
2. The input and output vectors are initialized in host memory.
3. The necessary amount of device memory for the input and output vectors is allocated and the input data is copied to the device.
4. A pointer to the device global variable `global` is obtained via `hipGetSymbolAddress`.
5. The pointee is initialized by copying a value from the host to it.
6. The device global variable `global_array` is initialized by copying to it directly with `hipMemcpyToSymbol`.
7. The GPU kernel is then launched with the previously defined arguments.
8. The results are copied back to the host.
9. Device memory backing the input and output vectors is freed.
10. A reference computation is performed on the host and the results are compared with the expected result. The result of the comparison is printed to standard output.

## Key APIs and Concepts

Apart from via kernel parameters, values can also be passed to the device via _device global variables_: global variables that have the `__device__` attribute. These can be used from device kernels, and need to be initialized from the host before they hold a valid value. Device global variables are persistent between kernel launches, so they can also be used to communicate values between lauches without explicitly managing a buffer for the on the host.

A device global variable cannot be used as a regular global variable from the host side. To manage them, a pointer to the device memory that they represent needs to be obtained first. This can be done using the functions `hipGetSymbolAddress(dev_ptr, symbol)` and `hipGetSymbolSize(dev_ptr, symbol)`. A device global variable can be passed directly to this function by using the `HIP_SYMBOL(symbol)` macro. The resulting device pointer can be used in the same ways as memory obtained from `hipMalloc`, and so the corresponding value can be set by using `hipMemcpy`.

Device global variables may also be initialized directly by using the `hipMemcpyToSymbol(symbol, host_source, size_bytes, offset = 0, kind = hipMemcpyHostToDevice)`. This method omits having to fetch the pointer to the device global variable explicitly. Similarly, `hipMemcpyFromSymbol(host_dest, symbol, size_bytes, offset = 0, kind = hipMemcpyDeviceToHost)` can be used to copy from a device global variable back to the host.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `__global__`
- `__device__`
- `threadIdx`
- `blockDim`
- `blockIdx`

#### Host symbols

- `hipFree`
- `hipGetLastError`
- `hipGetSymbolAddress`
- `hipGetSymbolSize`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipMemcpyToSymbol`
- `hipStreamDefault`
- `HIP_SYMBOL`
