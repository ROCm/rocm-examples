# hipBLASLt Tensor Manipulation - Weight Matrix Swizzling with Padding

## Description

This example illustrates the use of the `hipBLASLt` tensor manipulation utilities for weight matrix swizzling with automatic padding.

The operation demonstrates the complete pipeline for optimizing weight matrices:

1. **Original Weight Matrix**: Standard row-major layout
2. **Padded Weight Matrix**: Expanded dimensions to meet swizzling alignment requirements
3. **Swizzled Weight Matrix**: Optimized memory layout for improved GPU performance

## Application flow

1. Create an original weight matrix with specified dimensions (18×34).
2. Initialize the matrix with sequential integer values for visualization.
3. Display the original weight matrix structure.
4. Calculate padding requirements based on swizzling parameters.
5. Apply padding to align matrix dimensions with swizzling constraints.
6. Display the padded weight matrix showing zero-padding effects.
7. Reshape the padded matrix into multi-dimensional tensor format.
8. Apply permutation operations to achieve swizzled memory layout.
9. Display the final swizzled weight matrix structure.
10. Demonstrate the complete transformation pipeline for GPU optimization.

## Key APIs and Concepts

- **Tensor Manipulation Utilities**:
  - `tensor_manipulation::tensor`: A generic tensor class for creating and manipulating multi-dimensional data.
  - `tensor_manipulation::pad_tensor()`: A function to pad a tensor to meet specific alignment requirements for swizzling.
  - `tensor_manipulation::permute_tensor()`: A function to reorder the dimensions of a tensor to achieve a swizzled memory layout.
  - `print_tensor_data_multi_dims()`: A utility for visualizing the contents of multi-dimensional tensors.
