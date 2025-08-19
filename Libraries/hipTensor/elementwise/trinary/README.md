# hipTensor Elementwise Trinary Operation

## Description

This example illustrates how to perform an elementwise trinary operation using the `hipTensor` library.

The operation calculates the following:

$D_{c,w,h} = \gamma \cdot C_{w,h,c} + (\alpha \cdot A_{w,h,c} + \beta \cdot B_{w,h,c})$

where:

- $\alpha$, $\beta$, and $\gamma$ are scalars
- $A$, $B$, and $C$ are tensors of dimensions $w \times h \times c$
- $D$ is a tensor of dimensions $c \times w \times h$

## Application flow

1. Check if F32 operations are supported on the current device.
2. Define data types and set up tensor data type descriptors.
3. Set scalar coefficient values.
4. Define tensor modes and extents.
5. Allocate host and device memory.
6. Initialize tensor data.
7. Create a hipTensor handle.
8. Create tensor descriptors.
9. Create the elementwise trinary operation descriptor.
10. Create an execution plan.
11. Execute the elementwise trinary operation.
12. Clean up all resources.

## Key APIs and Concepts

- **hipTensor Initialization**: The hipTensor library is initialized by creating a handle with `hiptensorCreate()` and released with `hiptensorDestroy()`.

- **Tensor Descriptors**:
  - `hiptensorCreateTensorDescriptor()`: Creates a descriptor for a tensor, defining its data type, dimensions, and strides.
  - `hiptensorDestroyTensorDescriptor()`: Frees the tensor descriptor.

- **Elementwise Trinary Operation Descriptor**:
  - `hiptensorCreateElementwiseTrinary()`: Creates a descriptor for an elementwise trinary operation, specifying the tensors, their modes, and the operation types.
  - `hiptensorDestroyOperationDescriptor()`: Frees the elementwise trinary operation descriptor.

- **Algorithm Selection and Execution Plan**:
  - `hiptensorCreatePlanPreference()`: Creates a preference object to guide the algorithm selection process.
  - `hiptensorCreatePlan()`: Creates an execution plan based on the operation descriptors and preferences.
  - `hiptensorDestroyPlanPreference()`: Frees the preference object.
  - `hiptensorDestroyPlan()`: Frees the execution plan.

- **Execution**:
  - `hiptensorElementwiseTrinaryExecute()`: Executes the elementwise trinary operation using the created plan.

- **Key Enumerations**:
  - `hiptensorDataType_t`: Defines the data type of tensors (e.g., `HIPTENSOR_R_32F` for single-precision).
  - `hiptensorComputeDescriptor_t`: Sets the precision for the computation (e.g., `HIPTENSOR_COMPUTE_32F`).
  - `hiptensorOperator_t`: Specifies tensor operations, such as `HIPTENSOR_OP_IDENTITY` and `HIPTENSOR_OP_ADD`.

## Demonstrated API Calls

### hipTensor

- `hiptensorCreate`
- `hiptensorCreateElementwiseTrinary`
- `hiptensorCreatePlan`
- `hiptensorCreatePlanPreference`
- `hiptensorCreateTensorDescriptor`
- `hiptensorDestroy`
- `hiptensorDestroyOperationDescriptor`
- `hiptensorDestroyPlan`
- `hiptensorDestroyPlanPreference`
- `hiptensorDestroyTensorDescriptor`
- `hiptensorElementwiseTrinaryExecute`
- `hiptensorLoggerSetMask`

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
- `HIPTENSOR_R_32F`
- `HIPTENSOR_COMPUTE_32F`
- `HIPTENSOR_OP_IDENTITY`
- `HIPTENSOR_OP_ADD`
- `HIPTENSOR_ALGO_DEFAULT`
