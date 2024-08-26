# rocFFT Mutli GPU Example (C++)

## Description

This example illustrates the use of rocFFT multi-GPU functionality. It shows how to use multiple GPUs with rocFFT by using `rocfft_brick` and `rocfft_field` to divide the work between multiple devices. At least requires rocm version 6.0.0.

### Application flow

1. Read in command-line parameters.
2. Check if there are two device with 3-D inputs.
3. Check if there device ids that do not exist.
4. Create a plan description for multi-GPU plan.
5. Define infield geometry for both gpus and add bricks.
6. Add infield to the plan description.
7. Allocate and initialize GPU input.
8. Define outfield geometry for both gpus and add bricks.
9. Add outfield to the plan description.
10. Allocate and initialize GPU output.
11. Create multi-gpu `rocFFT` plan with the created plan description.
12. Get execution information and allocate work buffer.
13. Execute multi-gpu plan.
14. Get results from the first device.
15. Destroy plan and free device memory.

### Command line interface

The application provides the following optional command line arguments:

- `-l` or `--length`. The 3-D FFT size separated by spaces. It default value is `8 8 8`.
- `-d` or `--devices`. The list of devices to use separated by spaces. It default value is `0 1`.

## Key APIs and Concepts

- rocFFT is initialized by calling `rocfft_setup()` and it is cleaned up by calling `rocfft_cleanup()`.
- rocFFT creates a plan with `rocfft_plan_create`. This function takes many of the fundamental parameters needed to specify a transform. The plan is then executed with `rocfft_execute` and destroyed with `rocfft_plan_destroy`.
- `rocfft_field` is used to hold data decomposition information which is then passed to a `rocfft_plan` via a `rocfft_plan_description`
- `rocfft_brick` is used to describe the data decomposition of fields
- To execute HIP functions on different gpus `hipSetDevice` can be used with the id of the gpu to switch beteen gpus.

## Demonstrated API Calls

### rocFFT

- `rocfft_array_type_complex_interleaved`
- `rocfft_brick_create`
- `rocfft_brick_destroy`
- `rocfft_cleanup`
- `rocfft_execute`
- `rocfft_execution_info_create`
- `rocfft_execution_info_destroy`
- `rocfft_execution_info_set_work_buffer`
- `rocfft_field_add_brick`
- `rocfft_field_create`
- `rocfft_placement_notinplace`
- `rocfft_plan_create`
- `rocfft_plan_description_add_infield`
- `rocfft_plan_description_add_outfield`
- `rocfft_plan_description_destroy`
- `rocfft_plan_description_create`
- `rocfft_plan_description_set_data_layout`
- `rocfft_plan_destroy`
- `rocfft_plan_get_work_buffer_size`
- `rocfft_precision_double`
- `rocfft_setup`
- `rocfft_transform_type_complex_forward`

### HIP runtime

- `hipFree`
- `hipGetDeviceCount`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipSetDevice`
