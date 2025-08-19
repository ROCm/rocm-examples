# hipTensor BF16 Data with FP32 Compute Bilinear Contraction

## Description

This example illustrates how to perform a bilinear tensor contraction operation using the `hipTensor` library with BFloat16 precision for data storage and FP32 for computation.

The operation calculates the following:

$C_{m,n,u,v} = \alpha \cdot A_{m,n,h,k} \cdot B_{u,v,h,k} + \beta \cdot C_{m,n,u,v}$

where:

- $\alpha$ and $\beta$ are scalars
- $A$ is a tensor of dimensions $m \times n \times h \times k$
- $B$ is a tensor of dimensions $u \times v \times h \times k$
- $C$ is a tensor of dimensions $m \times n \times u \times v$

## Application flow

1. Check if F32 operations are supported on the current device.
2. Define BFloat16 data types for tensors and FP32 for the compute type.
3. Set up hipTensor data type descriptors.
4. Set scalar coefficient values.
5. Call the `bilinear_contraction_sample` template function.
6. Inside the template function:
    - Define tensor modes and extents.
    - Allocate host and device memory.
    - Initialize tensor data.
    - Create a hipTensor handle.
    - Create tensor descriptors.
    - Create the contraction operation descriptor.
    - Create an execution plan.
    - Execute the bilinear contraction.
    - Clean up all resources.

## Key APIs and Concepts

- **hipTensor Initialization**: The hipTensor library is initialized by creating a handle with `hiptensorCreate()` and released with `hiptensorDestroy()`.

- **Tensor Descriptors**:
  - `hiptensorCreateTensorDescriptor()`: Creates a descriptor for a tensor, defining its data type, dimensions, and strides.
  - `hiptensorDestroyTensorDescriptor()`: Frees the tensor descriptor.

- **Contraction Operation Descriptor**:
  - `hiptensorCreateContraction()`: Creates a descriptor for the contraction operation, specifying the tensors, their modes, and the computation type.
  - `hiptensorDestroyOperationDescriptor()`: Frees the contraction operation descriptor.

- **Algorithm Selection and Execution Plan**:
  - `hiptensorCreatePlanPreference()`: Creates a preference object to guide the algorithm selection process.
  - `hiptensorEstimateWorkspaceSize()`: Queries for the required workspace memory size for the contraction.
  - `hiptensorCreatePlan()`: Creates an execution plan based on the operation descriptors and preferences.
  - `hiptensorDestroyPlanPreference()`: Frees the preference object.
  - `hiptensorDestroyPlan()`: Frees the execution plan.

- **Execution**:
  - `hiptensorContract()`: Executes the tensor contraction using the created plan.

- **Key Enumerations**:
  - `hiptensorDataType_t`: Defines the data type of tensors (e.g., `HIPTENSOR_R_16BF` for BFloat16).
  - `hiptensorComputeDescriptor_t`: Sets the precision for the computation (e.g., `HIPTENSOR_COMPUTE_32F`).
  - `hiptensorOperator_t`: Specifies tensor operations, such as `HIPTENSOR_OP_IDENTITY`.

## Demonstrated API Calls

### hipTensor

- `hiptensorContract`
- `hiptensorCreate`
- `hiptensorCreateContraction`
- `hiptensorCreatePlan`
- `hiptensorCreatePlanPreference`
- `hiptensorCreateTensorDescriptor`
- `hiptensorDestroy`
- `hiptensorDestroyOperationDescriptor`
- `hiptensorDestroyPlan`
- `hiptensorDestroyPlanPreference`
- `hiptensorDestroyTensorDescriptor`
- `hiptensorEstimateWorkspaceSize`
- `hiptensorLoggerSetMask`

### HIP runtime

- `hipDeviceSynchronize`
- `hipFree`
- `hipHostMalloc`
- `hipMalloc`
- `hipMemcpy`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`

### Data Types and Enums

- `hiptensorHandle_t`
- `hiptensorTensorDescriptor_t`
- `hiptensorOperationDescriptor_t`
- `hiptensorPlan_t`
- `hiptensorPlanPreference_t`
- `hiptensorDataType_t`
- `hiptensorComputeDescriptor_t`
- `HIPTENSOR_R_16BF`
- `HIPTENSOR_COMPUTE_32F`
- `HIPTENSOR_OP_IDENTITY`
- `hip_bfloat16`
