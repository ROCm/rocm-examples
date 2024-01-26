# rocFFT callback Example (C++)

## Description
This example illustrates the use of rocFFT `callback` functionality. It shows how to use load callback, a user-defined callback function that is run to load input from global memory at the start of the transform, with rocFFT.

### Application flow
1. Set up rocFFT.
2. Allocate and initialize the host data and filter.
3. Allocate device memory.
4. Copy data and filter from host to device.
5. Create an FFT plan.
6. Check if FFT plan requires a work buffer, if true:
   - Allocate and set work buffer on device.
7. Allocate and initialize callback data on host.
8. Copy callback data from host to device.
9. Get a host pointer to the callback device function.
10. Set the callback with the callback data and device function.
11. Execute FFT plan which multiplies each element by filter element and scales.
12. Clean up work buffer and FFT plan.
13. Copy the results from device to host.
14. Print results.
15. Free device memory.
16. The cleanup of the rocFFT enviroment.

## Key APIs and Concepts
- rocFFT is initialized by calling `rocfft_setup()` and it is cleaned up by calling `rocfft_cleanup()`.
- rocFFT creates a plan with `rocfft_plan_create`. This function takes many of the fundamental parameters needed to specify a transform. The plan is then executed with `rocfft_execute` and destroyed with `rocfft_plan_destroy`.
- rocFFT can add work buffers and can control plan execution with `rocfft_execution_info` from `rocfft_execution_info_create(rocfft_execution_info *info)`. For this example specifically a load callback with `rocfft_execution_info_set_load_callback` and work buffer with `rocfft_execution_info_set_work_buffer`.
- [Callbacks](https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html#load-and-store-callbacks) is an experimental functionality in rocFFT. It requires a pointer to the shared memory, but did not support shared memory when this example was created.


## Demonstrated API Calls
### rocFFT
- `rocfft_cleanup`
- `rocfft_execute`
- `rocfft_execution_info_create`
- `rocfft_execution_info_destroy`
- `rocfft_execution_info_set_load_callback`
- `rocfft_execution_info_set_work_buffer`
- `rocfft_placement_inplace`
- `rocfft_plan_create`
- `rocfft_plan_destroy`
- `rocfft_plan_get_work_buffer_size`
- `rocfft_precision_double`
- `rocfft_setup`
- `rocfft_transform_type_complex_forward`

### HIP runtime
- `HIP_SYMBOL`
- `hipCmul`
- `hipFree`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyFromSymbol`
- `hipMemcpyHostToDevice`
- `make_hipDoubleComplex`
