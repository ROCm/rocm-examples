# rocPRIM Block Sum Example

## Description

This simple program showcases the usage of the `rocprim::block_reduce` block-level function. It also showcases the usage of `rocprim::block_load` block-level load function. The results from `rocprim::block_load` are eventually used by `rocprim::block_reduce`. The final result of the block-level reductions are written to the standard output.

### Application flow

1. Host side data is instantiated in a `std::vector<int>`.
2. Device storage for input and output data is allocated using `hipMalloc`.
3. Input data is copied from the host to the device using  `hipMemcpy`.
4. Device kernel `reduce_sum_kernel` is launched using the `myKernelName<<<...>>>`-syntax.
    - The kernel uses `rocprim::block_load` to load input from the device global memory into per-thread local register memory.
    - The kernel uses `rocprim::block_reduce` to perform reduction on `valid_items` elements per block.
    - If the input array is not evenly divisible by the number of threads in a block then for that block the kernel sets the `valid_items` to the correct size, i.e. `valid_items = input_size % (BlockSize * ItemsPerThread);`
5. The result of the summation is copied back to the host and is printed to the standard output.
6. All device memory is freed using `hipFree`.

## Key APIs and Concepts

- rocPRIM provides HIP parallel primitives on multiple levels of the GPU programming model. This example showcases `rocprim::block_reduce` which is a GPU block-level function.
- The `rocprim::block_reduce` template function performs a reduction, i.e. it combines a vector of values to a single value using the provided binary operator. Since the order of execution is not determined, the provided operator must be associative. In the example, an addition (`rocprim::plus<int>`) is used, which fulfils this property.
- `rocprim::block_reduce` is a collective operation, which means all threads in the block must make a call to `rocprim::block_reduce`.
- In this example `rocprim::block_load` is used to pre-fetch (load) the global input data. It has the potential to increase performance since data is effiently loaded into per-thread local register space.

## Used API surface

### rocPRIM

- `rocprim::block_reduce`
- `rocprim::plus`
- `rocprim::block_load`

### HIP runtime

- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipFree`
