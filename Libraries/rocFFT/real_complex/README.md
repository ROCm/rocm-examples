# rocFFT Real to Complex Example (C++)

## Description

This example illustrates the use of rocFFT for a Fast Fourier Transform from real to complex numbers. It supports 1D, 2D and 3D transforms. This example can be both run on Windows and Linux. The sizes for the transform are as follows:

- 1D: $N_1$ to $\lfloor \frac{N_1}{2}  \rfloor + 1$
- 2D: $N_1 N_2$ to $(\lfloor \frac{N_1}{2}  \rfloor + 1) N_2$
- 3D: $N_1 N_2 N_3$ to $(\lfloor \frac{N_1}{2}  \rfloor + 1) N_2 N_3$

$N_1$, $N_2$ and $N_3$ are the size of the input, specified by the --length parameter of the example.

### Application flow

1. Read in command-line parameters.
2. Setup buffer size and stride for input and output.
3. Allocate, initialize and print host input data.
4. Allocate device memory and copy host input data to device.
5. Create a plan description and set the data layout.
6. Create a rocFFT plan with the previous plan description.
7. Get execution information and allocate work buffer.
8. Execute rocFFT plan.
9. Copy back output data to the host and print it.
10. Destroy plan and free device memory.

### Command line interface

The application provides the following optional command line arguments:

- `-d` or `--device`. The id of the device to be used. Its default value is `0`.
- `-o` or `--outofplace`. A boolean flag that if set the output will be written to another output buffer, if not set it will overwrite the input buffer. Its default value is `false`.
- `-l` or `--length`. The FFT size separated by spaces, it supports from 1 up to 3 dimensions. Its default value is `4 4`.

## Key APIs and Concepts

- rocFFT is initialized by calling `rocfft_setup()` and it is cleaned up by calling `rocfft_cleanup()`.
- rocFFT creates a plan with `rocfft_plan_create`. This function takes many of the fundamental parameters needed to specify a transform. The plan is then executed with `rocfft_execute` and destroyed with `rocfft_plan_destroy`.
- rocFFT supports both in-place and not-inplace result placements. With in-place only the input buffer is required and will be overwritten with the result. These result placement types are `rocfft_placement_inplace` and `rocfft_placement_inplace`.
- rocFFT transforms types can be given to the set data layout for the plan description. This example the input data format is, `rocfft_array_type_real`, and the output data format is, `rocfft_array_type_hermitian_interleaved`.

## Demonstrated API Calls

### rocFFT

- `rocfft_array_type_hermitian_interleaved`
- `rocfft_array_type_real`
- `rocfft_cleanup`
- `rocfft_execute`
- `rocfft_execution_info_create`
- `rocfft_execution_info_destroy`
- `rocfft_execution_info_set_work_buffer`
- `rocfft_placement_inplace`
- `rocfft_placement_notinplace`
- `rocfft_plan_create`
- `rocfft_plan_description_create`
- `rocfft_plan_description_destroy`
- `rocfft_plan_description_set_data_layout`
- `rocfft_plan_destroy`
- `rocfft_plan_get_work_buffer_size`
- `rocfft_precision_double`
- `rocfft_setup`
- `rocfft_transform_type_real_forward`

### HIP runtime

- `hipFree`
- `hipGetErrorString`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipSetDevice`
