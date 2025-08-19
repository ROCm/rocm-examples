# hipTensor Tensor Reduction

## Description

This example illustrates how to perform a tensor reduction operation using the `hipTensor` library.

The operation calculates the following:

$C_{k,v} = \alpha \cdot \sum_{m,h} A_{m,h,k,v} + \beta \cdot C_{k,v}$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a tensor of dimensions $m \times h \times k \times v$
- $C$ is a tensor of dimensions $k \times v$

## Application flow

1. Check if F32 operations are supported on the current device.
2. Define data types and set up tensor data type descriptors.
3. Set scalar coefficient values.
4. Define tensor modes and extents.
5. Allocate host and device memory.
6. Initialize tensor data.
7. Create a hipTensor handle.
8. Create tensor descriptors.
9. Create the reduction operation descriptor.
10. Create an execution plan.
11. Execute the reduction operation.
12. Clean up all resources.

## Key APIs and Concepts

- **hipTensor Initialization**: The hipTensor library is initialized by creating a handle with `hiptensorCreate()` and released with `hiptensorDestroy()`.

- **Tensor Descriptors**:
  - `hiptensorCreateTensorDescriptor()`: Creates a descriptor for a tensor, defining its data type, dimensions, and strides.
  - `hiptensorDestroyTensorDescriptor()`: Frees the tensor descriptor.

- **Reduction Operation Descriptor**:
  - `hiptensorCreateReduction()`: Creates a descriptor for a reduction operation, specifying the input and output tensors, their modes, and the reduction operator.
  - `hiptensorDestroyOperationDescriptor()`: Frees the reduction operation descriptor.

- **Algorithm Selection and Execution Plan**:
  - `hiptensorCreatePlanPreference()`: Creates a preference object to guide the algorithm selection process.
  - `hiptensorEstimateWorkspaceSize()`: Queries for the required workspace memory size for the reduction.
  - `hiptensorCreatePlan()`: Creates an execution plan based on the operation descriptors and preferences.
  - `hiptensorDestroyPlanPreference()`: Frees the preference object.
  - `hiptensorDestroyPlan()`: Frees the execution plan.

- **Execution**:
  - `hiptensorReduce()`: Executes the tensor reduction using the created plan.

- **Key Enumerations**:
  - `hiptensorDataType_t`: Defines the data type of tensors (e.g., `HIPTENSOR_R_32F` for single-precision).
  - `hiptensorComputeDescriptor_t`: Sets the precision for the computation (e.g., `HIPTENSOR_COMPUTE_32F`).
  - `hiptensorOperator_t`: Specifies tensor operations, such as `HIPTENSOR_OP_IDENTITY` and `HIPTENSOR_OP_ADD`.

## Demonstrated API Calls

### hipTensor

- `hiptensorCreate`
- `hiptensorCreatePlan`
- `hiptensorCreatePlanPreference`
- `hiptensorCreateReduction`
- `hiptensorCreateTensorDescriptor`
- `hiptensorDestroy`
- `hiptensorDestroyOperationDescriptor`
- `hiptensorDestroyPlan`
- `hiptensorDestroyPlanPreference`
- `hiptensorDestroyTensorDescriptor`
- `hiptensorEstimateWorkspaceSize`
- `hiptensorLoggerSetMask`
- `hiptensorReduce`

### HIP runtime

- `hipFree`
- `hipHostFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpy`

### Data Types and Enums

- `hiptensorHandle_t`
- `hiptensorTensorDescriptor_t`
- `hiptensorOperationDescriptor_t`
- `hiptensorPlan_t`
- `hiptensorPlanPreference_t`
- `hiptensorDataType_t`
- `hiptensorComputeDescriptor_t`
- `hiptensorAlgo_t`
- `hiptensorWorksizePreference_t`
- `HIPTENSOR_R_32F`
- `HIPTENSOR_COMPUTE_32F`
- `HIPTENSOR_OP_IDENTITY`
- `HIPTENSOR_OP_ADD`
- `HIPTENSOR_ALGO_DEFAULT`
- `HIPTENSOR_WORKSPACE_DEFAULT`
