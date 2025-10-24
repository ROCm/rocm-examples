# hipSPARSE Handle Management

## Description

This example demonstrates basic hipSPARSE library initialization, handle management, and version querying.

The operation illustrates fundamental hipSPARSE library setup:

- Library initialization and cleanup
- Version information retrieval
- Git revision querying

## Application flow

1. Create a hipSPARSE handle using `hipsparseCreate()`.
2. Query the hipSPARSE library version using `hipsparseGetVersion()`.
3. Query the hipSPARSE Git revision string using `hipsparseGetGitRevision()`.
4. Display the version information in human-readable format.
5. Clean up the hipSPARSE handle using `hipsparseDestroy()`.

## Key APIs and Concepts

- **hipSPARSE Handle Management**: The hipSPARSE library uses a handle-based design pattern where all operations require a valid handle.
  - `hipsparseCreate()`: Initializes the hipSPARSE library and creates a handle that maintains library context and state.
  - `hipsparseDestroy()`: Releases all resources associated with the hipSPARSE handle and shuts down the library context.

- **Version Information**: hipSPARSE provides APIs to query library version and build information.
  - `hipsparseGetVersion()`: Returns the library version as an integer that can be decoded into major, minor, and patch versions.
  - The version integer format: `(major * 100000) + (minor * 1000) + patch`
  - Example: Version 5.2.1 would be encoded as `5 * 100000 + 2 * 1000 + 1 = 502001`

- **Git Revision Tracking**:
  - `hipsparseGetGitRevision()`: Returns the Git commit hash used to build the library.
  - Useful for debugging and ensuring reproducibility across different builds.
  - Returns a null-terminated string that can be directly printed.

- **Resource Management**: Proper handle management is crucial for:
  - Memory leak prevention
  - Correct library initialization
  - Thread-safe operation in multi-threaded environments

## Demonstrated API Calls

### hipSPARSE

- `hipsparseCreate`
- `hipsparseDestroy`
- `hipsparseGetVersion`
- `hipsparseGetGitRevision`

### Data Types and Enums

- `hipsparseHandle_t`
