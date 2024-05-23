# rocThrust Device Pointer Example

## Description

This simple program showcases the usage of the `thrust::device_ptr` template.

### Application flow

1. A `thrust::device_ptr<int>` is instantiated, and memory for ten elements is allocated.
2. Two more `thrust::device_ptr<int>` are instantiated and set to the start- and end-point of the allocated memory region.
3. Normal pointer arithmetic is used on the `thrust::device_ptr<int>`s to calculate the number of elements allocated in step 1.
4. The elements pointed to by `thrust::device_ptr<int>` are initialized using `thrust::sequence`. The values of the elements are printed to the standard output.
5. The first three elements are modified directly in host-code with `thrust::device_ptr::operator[]` and all elements are printed to the standard output.
6. The raw pointer to the device memory is obtained by calling `thrust::device_pointer_cast`
7. The raw pointer is wrapped back to a `thrust::device_ptr<int>`. It is asserted that the original and the wrapped `thrust::device_ptr<int>` are equal.
8. The sum of the wrapped pointer is calculated with `thrust::reduce` and printed to the standard output.
9. The device memory is freed using `thrust::device_free`.

## Key APIs and Concepts

- Thrust's `device_ptr` is a simple and transparent way of handling device memory the same way one would handle host memory with normal pointers.
- Unlike a normal pointer to device memory `device_ptr` adds type safety, and the underlying device memory is transparently accessible on the host.
- The `device_ptr` can be used in Thrust algorithms like a normal pointer to device memory.
- The "raw" normal pointer to the device memory for usage in kernels or other APIs can be obtained from a `device_ptr` by using `thrust::raw_pointer_cast`.
- `device_ptr` is not a smart pointer. Allocating and freeing memory lies in the responsibility of the programmer.

## Demonstrated API Calls

### rocThrust

- `thrust::device_ptr<T>::operator=`
- `thrust::device_ptr<T>::operator[]`
- `thrust::device_malloc<T>`
- `thrust::sequence`
- `thrust::raw_pointer_cast`
- `thrust::device_pointer_cast`
- `thrust::reduce`
- `thrust::device_free`
