# HIP-Doc Address Retrieval Example

## Description

This example demonstrates how to retrieve the address of a HIP runtime function. For more information on this topic,
please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_driver_api.html#address-retrieval).

### Application flow

1. The HIP runtime is initialized in the usual way by through the HIP runtime function `hipInit`.
2. The address of the `hipInit` runtime function is obtained and assigned to a function pointer.
3. The HIP runtime is initialized again using the function pointer.

## Key APIs and Concepts

* `hipGetProcAddress` loads the address of a HIP runtime function and assigns it to a function pointer.

## Demonstrated API calls

### HIP runtime

#### Host symbols

* `hipGetProcAddress`
* `hipInit`
