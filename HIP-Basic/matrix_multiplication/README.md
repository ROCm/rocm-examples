# HIP-Basic Matrix Multiplication Example

## Description

This example showcases the multiplication of two dynamically sized two-dimensional matrices on the GPU ($\mathrm{A \cdot B=C}$). The sizes of the matrices can be provided on the command line, however the sizes must be multiples of the hard-coded block size, which is 16x16. This implementation is not aimed at best performance or best generality, although some optimizations, such as the utilization of shared memory, are in place.

### Application flow

1. Default values for dimensions of matrix $\mathrm{A}$ and the number of columns of matrix $\mathrm{B}$ are set.
2. Command line arguments are parsed (if any) and the matrix dimensions are updated. If the command line arguments do not match the specification, an error message is printed to the standard output and the program terminates with a non-zero exit code.
3. Host memory is allocated for the matrices $\mathrm{A}$, $\mathrm{B}$ and $\mathrm{C}$ (using `std::vector<float>`) and the elements of both $\mathrm{A}$ and $\mathrm{B}$ are set to two different constant values.
4. Device memory is allocated for all matrices and the elements of $\mathrm{A}$ and $\mathrm{B}$ are copied to the device.
5. The dimensions of the kernel grid is calculated based on the matrix dimensions. The matrix multiplication kernel is queued to the default stream.
6. The elements of the resulting matrix $\mathrm{C}$ are copied to the host and all device memory is freed.
7. The elements of $\mathrm{C}$ are compared with the expected result. The result of the comparison is printed to the standard output.

### Command line interface

- If no command line argument is provided, the default matrix sizes are used.

- Otherwise, exactly 3 arguments must be provided. All must be positive integers which are multiples of the block size (16). The order of the arguments is the following: rows of $\mathrm{A}$, columns of $\mathrm{A}$, columns of $\mathrm{B}$. Notice that rows of $\mathrm{B}$ cannot be specified, as it must match the columns of $\mathrm{A}$.

## Key APIs and Concepts

- The kernel implemented in this example performs a matrix multiplication over dynamically sized matrices. The value of $\mathrm{C}$ at row $i$ and column $j$ is calculated with the following formula (where $N$ equals to the columns of $\mathrm{A}$ and rows of $\mathrm{B}$):

$$c_{ij}=\sum_{k=1}^{N}a_{ik}b_{kj}$$

- The kernel is launched in a two-dimensional grid in which each thread is responsible for calculating a single element of the resulting matrix. The threads are organized into 16x16 blocks. Since each block is executed on a single compute unit of the GPU hardware, data can be exchanged between these threads via shared memory.

- The matrix multiplication is conducted in multiple steps, each step calculating the partial results of a submatrix of size 16x16 (the block size). The number of steps is the columns of $\mathrm{A}$ divided by the block size.

- For improved performance, in each step the threads first load the corresponding submatrices from both $\mathrm{A}$ and $\mathrm{B}$ to the shared memory. Thereby each thread has to perform only one global memory fetch instead of loading the full 16 item row from each submatrix.

- Between loading and using values to/from shared memory, a call to `__syncthreads` has to be invoked. This is to ensure that all threads have finished writing to the shared memory before other threads might use the same memory locations.

  - The reason behind this is that it is not guaranteed that all threads in the block execute concurrently. Indeed, the compute unit schedules the threads to execute in so called "wavefronts". While one wavefront is waiting for memory operations to complete, another one might get scheduled to execute. The call to `__syncthreads` ensures that all threads in the block finish the pending memory operations and the loaded memory can safely be used from any other thread.

## Used API surface

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__shared__`
- `__syncthreads`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
