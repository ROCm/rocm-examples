# hipCUB Device Sum Example

## Description

This simple program showcases the usage of the `hipcub::DeviceReduce::Sum()`.

### Application flow

1. Host side data is instantiated in a `std::vector<int>`.
2. Device side storage is allocated using `hipMalloc`.
3. Data is copied from host to device using `hipMemcpy`.
4. `hipCUB` makes use of some temporary memory on the device and that needs to be allocated. A first call to `hipcub::DeviceReduce::Sum` is made (and since `d_temp_storage` is set to null) it stores the size in bytes of temporary storage needed in `temp_storage_bytes`.
5. `temp_storage_bytes` is used to allocate device memory in `d_temp_storage` using `hipMalloc`.
6. Finally a call to `hipcub::DeviceReduce::Sum` is made that computes the sum.
7. Result is transfered from device to host.
8. Free any device side memory using `hipFree`

## Key APIs and Concepts

- hipCUB provided device level API is used in this example. It performs global device level operations (in this case a sum reduction using `hipcub::DeviceReduce::Sum`) on the GPU.

## Demonstrated API Calls

### hipCUB

- `hipcub::DeviceReduce::Sum`

### HIP runtime

- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipFree`
