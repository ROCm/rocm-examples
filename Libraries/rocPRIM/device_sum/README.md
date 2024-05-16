# rocPRIM Device Sum Example

## Description

This simple program showcases the usage of the device function `rocprim::reduce`.

### Application flow

1. Input data is instantiated in a `std::vector<int>` and the values are printed to the standard output.
2. Device storage for input and output data is allocated using `hipMalloc`.
3. Input data is copied from the host to the device using `hipMemcpy`.
4. The binary operator used in the reduction is instantiated. This example calculates the sum of the elements of the input vector, hence `rocprim::plus<int>` is the appropriate choice.
5. The amount of working memory needed by the reduction algorithm is calculated by a first call to `rocprim::reduce`. For the first argument, a `nullptr` is passed, thereby the function calculates the value of `temp_storage_bytes` and returns without launching the GPU kernel.
6. `temp_storage_bytes` amount of memory is allocated on the device.
7. A subsequent call to `rocprim::reduce` is made, this time passing the pointer to the working memory. This launches the GPU kernel that performs the calculation.
8. The result of the summation is copied back to the host and is printed to the standard output.
9. All device memory is freed using `hipFree`.

## Key APIs and Concepts

- rocPRIM provides HIP parallel primitives on multiple levels of the GPU programming model. This example showcases `rocprim::reduce` which is a device function, thereby it can be called from host code.
- The `rocprim::reduce` template function performs a generalized reduction, i.e. it combines a vector of values to a single value using the provided binary operator. Since the order of execution is not determined, the provided operator must be associative. In the example, an addition (`rocprim::plus<int>`) is used which fulfils this property.
- The device functions of `rocPRIM` require a temporary device memory location to store the results of intermediate calculations. The required amount of temporary storage can be calculated by invoking the function with matching argument set, except the first argument `temporary_storage` must be a `nullptr`. In this case, the GPU kernel is not launched.

## Demonstrated API Calls

### rocPRIM

- `rocprim::reduce`
- `rocprim::plus`

### HIP runtime

- `hipMalloc`
- `hipMemcpy`
- `hipFree`
