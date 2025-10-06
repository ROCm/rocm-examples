# MIGraphX Custom Operator with rocBLAS Kernel

## Description

This example demonstrates how to implement a custom operator using MIGraphX's C++ API with rocBLAS library calls. It shows how to integrate rocBLAS's optimized linear algebra routines into a MIGraphX program, allowing you to leverage rocBLAS's high-performance BLAS operations while seamlessly combining them with built-in MIGraphX operators.

## Application flow

1. Parse command line arguments for device ID, vector size, and scale factor.
2. Set the HIP device for GPU operations.
3. Define and register a custom operator that implements vector scaling using rocBLAS's `sscal` function.
4. Build a MIGraphX program that combines the custom operator with built-in operators (neg, relu).
5. Compile the program for GPU target with offload copy enabled.
6. Prepare input data with sequential values and scale parameter.
7. Execute the program with the custom rocBLAS kernel.
8. Verify the output against expected results.

## Key APIs and Concepts

- **Custom Operator Definition**: MIGraphX allows extending functionality through custom operators.
  - `migraphx::experimental_custom_op_base`: Base class for implementing custom operators.
  - `name()`: Returns the unique identifier for the custom operation.
  - `runs_on_offload_target()`: Indicates whether the operation runs on GPU (true) or CPU (false).
  - `compute()`: Implements the actual computation using rocBLAS library calls.
  - `compute_shape()`: Validates input shapes and determines the output shape.

- **Custom Operator Registration**:
  - `migraphx::register_experimental_custom_op()`: Registers a custom operator with MIGraphX, making it available for use in programs.

- **rocBLAS Integration**: rocBLAS provides optimized BLAS (Basic Linear Algebra Subprograms) operations.
  - `rocblas_create_handle()`: Creates a rocBLAS handle for library operations.
  - `rocblas_set_stream()`: Associates a rocBLAS handle with a HIP stream.
  - `rocblas_set_pointer_mode()`: Configures whether scalar parameters are on host or device.
  - `rocblas_sscal()`: Scales a vector by a scalar (single-precision: `x = alpha * x`).
  - `rocblas_destroy_handle()`: Destroys a rocBLAS handle and frees resources.
  - `ctx.get_queue<hipStream_t>()`: Retrieves the HIP stream from the MIGraphX context.

- **Program Construction**:
  - `migraphx::program`: Container for the computational graph.
  - `get_main_module()`: Retrieves the main module for building the graph.
  - `add_parameter()`: Adds an input parameter to the program.
  - `add_instruction()`: Adds an operation to the graph.
  - `add_return()`: Specifies the program's output.
  - `migraphx::operation()`: Creates an operation object by name.

- **Compilation and Execution**:
  - `migraphx::compile_options`: Configuration for compilation.
  - `set_offload_copy()`: Enables automatic memory management for GPU operations.
  - `compile()`: Compiles the program for the specified target.
  - `eval()`: Executes the compiled program with provided parameters.

## Demonstrated API Calls

### MIGraphX

- `migraphx::experimental_custom_op_base`
- `migraphx::register_experimental_custom_op`
- `migraphx::program`
- `migraphx::program::get_main_module`
- `migraphx::module::add_parameter`
- `migraphx::module::add_instruction`
- `migraphx::module::add_return`
- `migraphx::operation`
- `migraphx::compile_options`
- `migraphx::compile_options::set_offload_copy`
- `migraphx::program::compile`
- `migraphx::target`
- `migraphx::program_parameters`
- `migraphx::program_parameters::add`
- `migraphx::argument`
- `migraphx::shape`
- `migraphx::program::eval`
- `migraphx::context::get_queue`

### rocBLAS

- `rocblas_create_handle`
- `rocblas_set_stream`
- `rocblas_set_pointer_mode`
- `rocblas_sscal`
- `rocblas_destroy_handle`
- `rocblas_status_to_string`

### HIP Runtime

- `hipSetDevice`
- `hipStream_t`

### Data Types and Enums

- `migraphx::program`
- `migraphx::module`
- `migraphx::shape`
- `migraphx::shapes`
- `migraphx::argument`
- `migraphx::arguments`
- `migraphx::context`
- `migraphx::compile_options`
- `migraphx::target`
- `migraphx::program_parameters`
- `migraphx_shape_float_type`
- `rocblas_handle`
- `rocblas_status`
- `rocblas_int`
- `rocblas_pointer_mode_device`
- `rocblas_status_success`
