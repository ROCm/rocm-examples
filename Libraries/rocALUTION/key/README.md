# rocALUTION Matrix Key Computation

## Description

This example demonstrates the use of the `rocALUTION` library for computing hash keys for matrix components.

The operation computes hash keys for:

- Row indices of matrix non-zero elements
- Column indices of matrix non-zero elements
- Values of matrix non-zero elements

This example shows how to generate hash keys that can be used for efficient matrix storage, indexing, or identification purposes. The key computation provides a way to uniquely identify matrix structures and can be useful for caching, debugging, or matrix comparison operations.

## Application flow

1. Parse command line arguments for matrix file path.
2. Initialize the rocALUTION platform.
3. Create rocALUTION objects and read the matrix from MTX format file.
4. Print matrix information including dimensions and non-zero count.
5. Compute hash keys for row indices, column indices, and values.
6. Report the computed keys for each matrix component.
7. Stop the rocALUTION platform.

## Key APIs and Concepts

- **rocALUTION Platform Management**: The rocALUTION library is initialized with `rocalution::init_rocalution()` and terminated with `rocalution::stop_rocalution()`. Platform information can be obtained using `rocalution::info_rocalution()`.

- **Matrix Operations**:
  - `rocalution::LocalMatrix::ReadFileMTX()`: Loads a matrix from Matrix Market format file.
  - `rocalution::LocalMatrix::Info()`: Prints matrix information including dimensions and non-zero count.
  - `rocalution::LocalMatrix::Key()`: Computes hash keys for row indices, column indices, and values.

- **Matrix Key Computation**:
  - **Hash Keys**: Generate unique identifiers for matrix components.
  - **Row Key**: Hash key computed from row indices of non-zero elements.
  - **Column Key**: Hash key computed from column indices of non-zero elements.
  - **Value Key**: Hash key computed from values of non-zero elements.
  - **Matrix Identification**: Keys can be used to uniquely identify matrix structure and content.

- **Hash Key Applications**:
  - **Caching**: Use keys for efficient matrix caching strategies.
  - **Comparison**: Compare matrices using their computed keys.
  - **Debugging**: Identify matrix modifications through key changes.
  - **Indexing**: Use keys for efficient sparse matrix data structures.

## Demonstrated API Calls

### rocALUTION Core Functions

- `rocalution::init_rocalution`
- `rocalution::stop_rocalution`

### Matrix Operations

- `rocalution::LocalMatrix::ReadFileMTX`
- `rocalution::LocalMatrix::Info`
- `rocalution::LocalMatrix::Key`

### Data Types

- `rocalution::LocalMatrix<double>`
- `long int`
