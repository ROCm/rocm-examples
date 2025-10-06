# MIGraphX Parse, Load, and Save Programs

## Description

This example demonstrates how to parse, load, and save graph programs using the MIGraphX C++ API. It shows how to work with ONNX models, serialize programs to MessagePack (.mxr) or JSON format, and load previously saved programs for execution.

## Application flow

1. Parse command line arguments to determine operation mode (parse/load) and file paths.
2. If parsing mode: Parse ONNX model file with specified batch size options.
3. If loading mode: Load previously saved program from MessagePack or JSON format.
4. Print the program's internal graph structure.
5. If save option specified: Serialize and save the program to the specified output file.
6. Display success message with operation details.

## Key APIs and Concepts

- **ONNX Parsing**: MIGraphX can parse ONNX models to create executable programs.
  - `migraphx::parse_onnx()`: Parses an ONNX model file into a MIGraphX program.
  - `migraphx::onnx_options`: Configuration object for ONNX parsing options.
  - `set_default_dim_value()`: Sets the default value for dynamic dimensions (e.g., batch size).

- **Program Serialization**: Programs can be saved to disk for later use.
  - `migraphx::save()`: Serializes a program to a file in MessagePack or JSON format.
  - `migraphx::file_options`: Configuration object for file operations.
  - `set_file_format()`: Specifies the serialization format ("msgpack" or "json").
  - MessagePack (.mxr) is the default binary format, offering compact storage.
  - JSON format provides human-readable serialization for debugging.

- **Program Loading**: Previously saved programs can be loaded without re-parsing.
  - `migraphx::load()`: Deserializes a program from a file.
  - Supports both MessagePack and JSON formats.
  - Loaded programs retain their compilation state if they were compiled before saving.

- **Program Inspection**:
  - `print()`: Outputs the program's internal graph structure for debugging and analysis.
  - Useful for understanding model architecture and verifying correct parsing.

## Demonstrated API Calls

### MIGraphX

- `migraphx::parse_onnx`
- `migraphx::onnx_options`
- `migraphx::onnx_options::set_default_dim_value`
- `migraphx::save`
- `migraphx::load`
- `migraphx::file_options`
- `migraphx::file_options::set_file_format`
- `migraphx::program::print`

### Data Types and Enums

- `migraphx::program`
- `migraphx::onnx_options`
- `migraphx::file_options`
