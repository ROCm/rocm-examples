# MIGraphX Custom Operator with MIOpen Kernel

## Description

This example demonstrates how to implement a custom operator using MIGraphX's C++ API with MIOpen library calls. It shows how to integrate MIOpen's optimized kernels into a MIGraphX program, allowing you to leverage MIOpen's deep learning primitives while seamlessly combining them with built-in MIGraphX operators.

## Application flow

1. Parse command line arguments for device ID and tensor dimensions.
2. Set the HIP device for GPU operations.
3. Define and register a custom operator that implements absolute value using MIOpen's activation function.
4. Build a MIGraphX program that combines the custom operator with built-in operators (neg, relu).
5. Allocate output buffer for the custom operator.
6. Compile the program for GPU target with offload copy enabled.
7. Prepare input data with sequential values.
8. Execute the program with the custom MIOpen kernel.
9. Verify the output against expected results.

## Key APIs and Concepts

- **Custom Operator Definition**: MIGraphX allows extending functionality through custom operators.
  - `migraphx::experimental_custom_op_base`: Base class for implementing custom operators.
  - `name()`: Returns the unique identifier for the custom operation.
  - `runs_on_offload_target()`: Indicates whether the operation runs on GPU (true) or CPU (false).
  - `compute()`: Implements the actual computation using MIOpen library calls.
  - `compute_shape()`: Validates input shapes and determines the output shape.

- **Custom Operator Registration**:
  - `migraphx::register_experimental_custom_op()`: Registers a custom operator with MIGraphX, making it available for use in programs.

- **MIOpen Integration**: MIOpen provides optimized deep learning primitives.
  - `miopenCreateWithStream()`: Creates a MIOpen handle associated with a HIP stream.
  - `miopenCreateTensorDescriptor()`: Creates a tensor descriptor for MIOpen operations.
  - `miopenSetTensorDescriptor()`: Configures tensor properties (data type, dimensions, strides).
  - `miopenCreateActivationDescriptor()`: Creates an activation function descriptor.
  - `miopenSetActivationDescriptor()`: Configures activation function parameters.
  - `miopenActivationForward()`: Executes the activation function on GPU.
  - `ctx.get_queue<hipStream_t>()`: Retrieves the HIP stream from the MIGraphX context.

- **Program Construction**:
  - `migraphx::program`: Container for the computational graph.
  - `get_main_module()`: Retrieves the main module for building the graph.
  - `add_parameter()`: Adds an input parameter to the program.
  - `add_instruction()`: Adds an operation to the graph.
  - `add_allocation()`: Allocates memory for intermediate results.
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
- `migraphx::module::add_allocation`
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

### MIOpen

- `miopenCreateWithStream`
- `miopenCreateTensorDescriptor`
- `miopenSetTensorDescriptor`
- `miopenCreateActivationDescriptor`
- `miopenSetActivationDescriptor`
- `miopenActivationForward`

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
- `migraphx_shape_half_type`
- `migraphx_shape_int32_type`
- `migraphx_shape_int8_type`
- `miopenHandle_t`
- `miopenTensorDescriptor_t`
- `miopenActivationDescriptor_t`
- `miopenActivationMode_t`
- `miopenDataType_t`
- `miopenFloat`
- `miopenHalf`
- `miopenInt32`
- `miopenInt8`
- `miopenActivationABS`
- `miopenStatusSuccess`
