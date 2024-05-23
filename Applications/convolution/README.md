# Applications Convolution Example

## Description

This example showcases a simple GPU implementation for calculating the [discrete convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution). The key point of this implementation is that in the GPU kernel each thread calculates the value for a convolution for a given element in the resulting grid.

For storing the mask constant memory is used. Constant memory is a read-only memory that is limited in size, but offers faster access times than regular memory. Furthermore on some architectures it has a separate cache. Therefore accessing constant memory can reduce the pressure on the memory system.

### Application flow

1. Default values for the size of the grid, mask and the number of iterations for the algorithm execution are set.
2. Command line arguments are parsed.
3. Host memory is allocated for the input, output and the mask. Input data is initialized with random numbers between 0-256.
4. Input data is copied to the device.
5. The simple convolution kernel is executed multiple times. Number of iterations is specified by the `-i` flag.
6. The resulting convoluted grid is copied to the host and device memory is freed.
7. The mean time in milliseconds needed for each iteration is printed to standard output as well as the mean estimated bandwidth.
8. The results obtained are compared with the CPU implementation of the algorithm. The result of the comparison is printed to the standard output.
9. In case requested the convoluted grid, the input grid, and the reference results are printed to standard output.

### Command line interface

There are three parameters available:

- `-h` displays information about the available parameters and their default values.
- `-x width` sets the grid size in the x direction. Default value is 4096.
- `-y height` sets the grid size in the y direction. Default value is 4096.
- `-p` Toggles the printing of the input, reference and output grids.
- `-i iterations` sets the number of times that the algorithm will be applied to the (same) grid. It must be an integer greater than 0. Its default value is 10.

## Key APIs and Concepts

- For this GPU implementation of the simple convolution calculation, the main kernel (`convolution`) is launched in a 2-dimensional grid. Each thread computes the convolution for one element of the resulting grid.

- Device memory is allocated with `hipMalloc` which is later freed by `hipFree`.

- Constant memory is declared in global scope for the mask, using the `__constant__` qualifier. The size of the object stored in constant memory must be available at compile time. Later the memory is initialized with `hipMemcpyToSymbol`.

- With `hipMemcpy` data can be transferred from host to device (using `hipMemcpyHostToDevice`) or from device to host (using `hipMemcpyDeviceToHost`).

- `myKernelName<<<...>>>` queues the kernel execution on the device. All the kernels are launched on the default stream `hipStreamDefault`, meaning that these executions are performed in order. `hipGetLastError` returns the last error produced by any runtime API call, allowing to check if any kernel launch resulted in an error.

- `hipEventCreate` creates the events used to measure kernel execution time, `hipEventRecord` starts recording an event and `hipEventSynchronize` waits for all the previous work in the stream when the specified event was recorded. These three functions can be used to measure the start and stop times of the kernel, and with `hipEventElapsedTime` the kernel execution time (in milliseconds) can be obtained. With `hipEventDestroy` the created events are freed.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `__global__`
- `__constant__`
- `hipEventCreate`
- `hipEventDestroy`
- `hipEventElapsedTime`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipMemcpyToSymbol`
- `hipStreamDefault`
