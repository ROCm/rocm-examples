# rocFFT callback Example (C++)

## Description

This example illustrates the use of rocFFT `callback` functionality. It shows how to use load callback, a user-defined callback function that is run to load input from global memory at the start of the transform, with rocFFT. Additionally, it shows how to make use of rocFFT's result scaling functionality.

### Application flow

1. Set up rocFFT.
2. Allocate and initialize the host data and filter.
3. Allocate device memory.
4. Compile the JIT load callback from source code.
5. Copy data and filter from host to device.
6. Allocate and initialize callback data on host.
7. Copy callback data from host to device.
8. Set up scaling factor and pass the compiled JIT load callback to rocFFT while creating an FFT plan.
9. Check if FFT plan requires a work buffer, if true:
   - Allocate and set work buffer on device.
10. Execute FFT plan which multiplies each element by filter element and scales.
11. Clean up work buffer and FFT plan.
12. Copy the results from device to host.
13. Print results.
14. Free device memory.
15. The cleanup of the rocFFT enviroment.

## Key APIs and Concepts

- rocFFT is initialized by calling `rocfft_setup()` and it is cleaned up by calling `rocfft_cleanup()`.
- This example compiles and sets a [load callback](https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html#load-and-store-callbacks) on a plan with `rocfft_plan_description_set_load_callback`.
- rocFFT creates a plan with `rocfft_plan_create`. This function takes many of the fundamental parameters needed to specify a transform. The plan is then executed with `rocfft_execute` and destroyed with `rocfft_plan_destroy`.
- rocFFT can add work buffers and can control plan execution with `rocfft_execution_info` from `rocfft_execution_info_create(rocfft_execution_info *info)`.  Data is passed to the callback function via `rocfft_execution_info_set_load_callback_data`.
- rocFFT provides explicit API for [result scaling](https://rocm.docs.amd.com/projects/rocFFT/en/latest/how-to/working-with-rocfft.html#result-scaling), which offers a more convenient way to perform this common operation than compiling and setting a callback function. The API exposed is `rocfft_plan_description_set_scale_factor`, which is to be used _before_ creating the plan.

## Demonstrated API Calls

### rocFFT

- `rocfft_cleanup`
- `rocfft_execute`
- `rocfft_execution_info_create`
- `rocfft_execution_info_destroy`
- `rocfft_execution_info_set_load_callback_data`
- `rocfft_execution_info_set_work_buffer`
- `rocfft_plan_create`
- `rocfft_plan_description`
- `rocfft_plan_description_create`
- `rocfft_plan_description_destroy`
- `rocfft_plan_description_set_load_callback`
- `rocfft_plan_description_set_scale_factor`
- `rocfft_plan_destroy`
- `rocfft_plan_get_work_buffer_size`
- `rocfft_setup`

### HIP runtime

- `hipCmul`
- `hipFree`
- `hipMalloc`
- `hipMemcpy`
- `make_hipDoubleComplex`
