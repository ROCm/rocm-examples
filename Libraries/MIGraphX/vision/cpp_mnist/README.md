# MIGraphX MNIST Inference

## Description

This example demonstrates how to perform inference on the MNIST handwritten digit dataset using the MIGraphX C++ API. It shows how to parse an ONNX model, apply optional quantization (FP16 or INT8), compile for different targets (CPU, GPU, or reference), and execute inference on randomly selected digits from the test set.

## Application flow

1. Parse command line arguments for model path, data file, target device, and quantization options.
2. Parse the ONNX model file into a MIGraphX program.
3. Create the target object for the specified device (cpu, gpu, or ref).
4. Apply optional quantization (FP16 or INT8 with optional calibration).
5. Compile the program for the target device with appropriate options.
6. Load a random digit from the test dataset.
7. Prepare input parameters with the digit data.
8. Execute the program and measure inference time.
9. Process the output to determine the predicted digit.
10. Verify the prediction against the actual digit label.

## Key APIs and Concepts

- **ONNX Model Parsing**: MIGraphX can parse ONNX models for inference.
  - `migraphx::parse_onnx()`: Parses an ONNX model file into a MIGraphX program.
  - `migraphx::onnx_options`: Configuration object for ONNX parsing options.

- **Quantization**: MIGraphX supports model quantization to reduce precision and improve performance.
  - `migraphx::quantize_fp16()`: Quantizes the program to 16-bit floating-point precision.
  - `migraphx::quantize_int8()`: Quantizes the program to 8-bit integer precision.
  - `migraphx::quantize_int8_options`: Configuration for INT8 quantization.
  - `add_calibration_data()`: Adds calibration data for more accurate INT8 quantization.
  - Quantization can improve performance and reduce memory usage with minimal accuracy loss.

- **Target Selection**: MIGraphX supports multiple execution targets.
  - `migraphx::target()`: Creates a target object for compilation.
  - "gpu": Compiles for AMD GPU execution with optimized kernels.
  - "cpu": Compiles for CPU execution with optimizations (requires `-DMIGRAPHX_ENABLE_CPU=On`).
  - "ref": Reference implementation primarily for correctness checking.

- **Compilation Options**:
  - `migraphx::compile_options`: Configuration for program compilation.
  - `set_offload_copy()`: Enables automatic memory transfers for GPU targets.
  - `set_fast_math()`: Enables fast math optimizations (may reduce accuracy slightly).
  - `compile()`: Compiles the program for the specified target.

- **Program Execution**:
  - `migraphx::program_parameters`: Container for input parameters.
  - `get_parameter_shapes()`: Retrieves the shapes of all input parameters.
  - `names()`: Returns the names of all parameters.
  - `add()`: Adds a named parameter with its argument data.
  - `migraphx::argument()`: Creates an argument from a shape and data pointer.
  - `eval()`: Executes the compiled program and returns outputs.

- **Output Processing**:
  - `get_shape()`: Retrieves the shape of an output tensor.
  - `lengths()`: Returns the dimensions of the tensor.
  - `data()`: Returns a pointer to the tensor data.
  - Results can be processed to extract predictions (e.g., argmax for classification).

- **Program Inspection**:
  - `print()`: Outputs the program's internal graph structure for debugging.

## Demonstrated API Calls

### MIGraphX

- `migraphx::parse_onnx`
- `migraphx::onnx_options`
- `migraphx::target`
- `migraphx::quantize_fp16`
- `migraphx::quantize_int8`
- `migraphx::quantize_int8_options`
- `migraphx::quantize_int8_options::add_calibration_data`
- `migraphx::compile_options`
- `migraphx::compile_options::set_offload_copy`
- `migraphx::compile_options::set_fast_math`
- `migraphx::program::compile`
- `migraphx::program::get_parameter_shapes`
- `migraphx::program::eval`
- `migraphx::program::print`
- `migraphx::program_parameters`
- `migraphx::program_parameters::add`
- `migraphx::parameter_shapes::names`
- `migraphx::argument`
- `migraphx::shape`
- `migraphx::shape::lengths`
- `migraphx::shape::elements`
- `migraphx::argument::get_shape`
- `migraphx::argument::data`

### Data Types and Enums

- `migraphx::program`
- `migraphx::onnx_options`
- `migraphx::target`
- `migraphx::compile_options`
- `migraphx::quantize_int8_options`
- `migraphx::program_parameters`
- `migraphx::parameter_shapes`
- `migraphx::argument`
- `migraphx::shape`
