# MIGraphX Dynamic Batch Processing

## Description

This example demonstrates how to run a graph program with dynamic batch sizes using the MIGraphX C++ API. Dynamic batch processing allows a single compiled model to handle inputs with varying batch dimensions at runtime, improving flexibility without requiring recompilation for each batch size.

## Application flow

1. Parse command line arguments for ONNX model path and batch size.
2. Set up dynamic dimensions with minimum, maximum, and optimal values.
3. Configure ONNX parsing options with dynamic input parameter shapes.
4. Parse the ONNX model file into a MIGraphX program.
5. Set up compilation options with offload copy enabled (required for dynamic batch).
6. Compile the program for GPU target.
7. Prepare input data with the specified batch size.
8. Create program parameters with input tensors.
9. Execute the program with the dynamic batch input.
10. Verify the output results against expected values.

## Key APIs and Concepts

- **Dynamic Dimensions**: MIGraphX uses `dynamic_dimension` objects to specify a range of dimension values that tensors can have at evaluation time.
  - `migraphx::dynamic_dimension`: Defines a dimension with minimum, maximum, and optional optimal values (e.g., `{min:1, max:10, optimals:{1, 4, 10}}`).
  - Fixed dimensions can be created by setting min and max to the same value.
  - Optimal values allow MIGraphX to optimize the program for specific shapes.

- **ONNX Parsing with Dynamic Shapes**:
  - `migraphx::parse_onnx()`: Parses an ONNX model file into a MIGraphX program.
  - `migraphx::onnx_options`: Configuration object for ONNX parsing.
  - `set_default_dyn_dim_value()`: Sets default dynamic dimension for symbolic batch variables.
  - `set_dyn_input_parameter_shape()`: Specifies the complete dynamic shape for a specific input parameter.

- **Program Compilation**:
  - `migraphx::compile_options`: Configuration object for compilation settings.
  - `set_offload_copy()`: Enables automatic memory copy operations for offloaded memory (required for dynamic batch on GPU).
  - `compile()`: Compiles the program for a specific target with given options.
  - `migraphx::target()`: Creates a target object (e.g., "gpu", "cpu", "ref").

- **Program Execution**:
  - `migraphx::program_parameters`: Container for input parameters.
  - `add()`: Adds a named parameter with its argument data.
  - `migraphx::argument()`: Creates an argument from a shape and data pointer.
  - `migraphx::shape()`: Defines tensor shape with data type and dimensions.
  - `eval()`: Executes the program with provided parameters and returns outputs.

- **Shape and Data Types**:
  - `migraphx_shape_uint8_type`: Unsigned 8-bit integer data type.
  - Dynamic shapes are defined by lists of `dynamic_dimension` objects.
  - Static shapes use fixed dimension values.

## Demonstrated API Calls

### MIGraphX

- `migraphx::parse_onnx`
- `migraphx::onnx_options`
- `migraphx::onnx_options::set_dyn_input_parameter_shape`
- `migraphx::dynamic_dimension`
- `migraphx::dynamic_dimensions`
- `migraphx::compile_options`
- `migraphx::compile_options::set_offload_copy`
- `migraphx::program::compile`
- `migraphx::target`
- `migraphx::program_parameters`
- `migraphx::program_parameters::add`
- `migraphx::argument`
- `migraphx::shape`
- `migraphx::program::eval`
- `migraphx::program::get_parameter_shapes`

### Data Types and Enums

- `migraphx::program`
- `migraphx::onnx_options`
- `migraphx::dynamic_dimension`
- `migraphx::dynamic_dimensions`
- `migraphx::compile_options`
- `migraphx::target`
- `migraphx::program_parameters`
- `migraphx::argument`
- `migraphx::shape`
- `migraphx_shape_uint8_type`
