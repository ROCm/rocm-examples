# rocThrust Vectors Example

## Description

This simple program showcases the usage of the `thrust::device_vector` and the `thrust::host_vector` templates.

### Application flow

1. A `thrust::host_vector<int>` is instantiated, its elements are set one-by-one, and the vector is printed to the standard output.
2. The `host_vector` is resized and it is printed again to the standard output.
3. A `thrust::device_vector<int>` is instantiated with the aforementioned `host_vector`. The contents are copied to the device.
4. The `device_vector`'s elements are modified from host code, and it is printed to the standard output.

## Key APIs and Concepts

- Thrust's device and host vectors implement RAII-style ownership over device and host memory pointers (similarly to `std::vector`). The instances are aware of the requested element count, allocate the required amount of memory, and free it upon destruction. When resized, the memory is reallocated if needed.
- Additionally, using `device_vector` and `host_vector` simplifies the transfers between device and host memory to a copy assignment.
- It is suggested that developers use `device_vector` and `host_vector` instead of explicit invocations to `malloc` and `free` functions.

## Demonstrated API Calls

### rocThrust

- `thrust::host_vector::host_vector`
- `thrust::host_vector::~host_vector`
- `thrust::host_vector::operator[]`
- `thrust::host_vector::resize()`
- `thrust::device_vector::device_vector`
- `thrust::device_vector::~device_vector`
- `thrust::device_vector::operator[]`
