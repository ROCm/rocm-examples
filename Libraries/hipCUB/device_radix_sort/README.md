# hipCUB Device Radix Sort Example

## Description

This simple program showcases the usage of the `hipcub::DeviceRadixSort::SortPairs` function.

### Application flow

1. Host side data is instantiated in `std::vector<float>` and `std::vector<int>` as key-value pairs.
2. Device side storage is allocated using `hipMalloc`.
3. Data is copied from host to device using `hipMemcpy`.
4. `hipCUB::DeviceRadixSort::SortPairs` makes use of temporary memory on the device that needs to be allocated manually. A first call to `hipcub::DeviceRadixSort::SortPairs` (with `d_temp_storage` set to null) calculates the size in bytes needed for the temporary storage and stores it in the variable `temp_storage_bytes`.
5. `temp_storage_bytes` is used to allocate device memory in `d_temp_storage` using `hipMalloc`.
6. Finally a second call to `hipcub::DeviceRadixSort::SortPairs` is invoked that sorts the pairs on the basis of keys.
7. Result is transferred from device to host.
8. Free all device side memory using `hipFree`

## Key APIs and Concepts

- The device-level API provided by hipCUB is used in this example. It performs global device level operations (in this case pair sorting using `hipcub::DeviceRadixSort::SortPairs`) on the GPU.

## Demonstrated API Calls

### hipCUB

- `hipcub::DoubleBuffer`
- `hipcub::DeviceRadixSort::SortPairs`

### HIP runtime

- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipFree`
