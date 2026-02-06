# HIP-Doc Histogram with Atomic Operations Example

## Description

This example demonstrates how to calculate an image histogram using HIP with
atomic operations. The application computes the brightness distribution of an
RGB image by averaging the color channels and binning the results. This example
showcases a fundamental GPU programming challenge: safely updating shared memory
locations from multiple threads.

The histogram calculation is a classic example of a race condition scenario in
parallel computing. Multiple threads may attempt to increment the same histogram
bin simultaneously, which could lead to incorrect results without proper
synchronization. This example uses HIP's atomic operations to ensure data
integrity.

For more information on atomic operations in HIP, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).

### Application flow

1. An RGB image is loaded from disk using the stb_image library.
2. The image data is normalized to the range [0, 1] for processing.
3. A histogram array is initialized with zero counts for all bins.
4. Device memory is allocated for the normalized image data and histogram.
5. The image data and initial histogram are copied to the device.
6. A 2D grid of thread blocks is configured based on image dimensions.
7. The histogram kernel is launched on the GPU where:
   - Each thread processes one pixel
   - Brightness is calculated by averaging RGB values
   - The appropriate bin is determined based on brightness
   - An atomic add operation safely increments the bin count
8. The kernel launch is checked for errors and synchronized.
9. The completed histogram is copied back to host memory.
10. The histogram is printed with statistics and validated.
11. All device memory is freed.

### Race Conditions and Atomic Operations

**The Problem**: In parallel computing, when multiple threads attempt to read,
modify, and write to the same memory location simultaneously, race conditions
can occur. For histogram calculation, many pixels may have similar brightness
values and map to the same bin, causing multiple threads to try incrementing
the same counter concurrently.

**The Solution**: Atomic operations provide a hardware-level guarantee that
read-modify-write sequences execute as indivisible operations. When a thread
performs `atomicAdd(&histogram[bin], 1)`, it:

1. Locks the memory location
2. Reads the current value
3. Adds 1 to it
4. Writes the result back
5. Releases the lock

This ensures correctness even when thousands of threads execute in parallel.

### Performance Considerations

While atomic operations ensure correctness, they can become a performance
bottleneck:

- **Serialization**: When multiple threads access the same memory address, they must wait in sequence, reducing parallelism
- **Memory Contention**: High contention on popular bins (e.g., images with uniform brightness) can significantly slow execution
- **Limited Parallelism**: The more threads compete for the same bins, the less benefit from GPU parallelization

**Potential Optimizations** (not implemented in this basic example):

- Use per-block shared memory histograms, then merge
- Privatization: Each thread block maintains its own histogram
- Interleaved binning to reduce bank conflicts
- Coarse-grained parallelism to reduce atomic contention

## Key APIs and Concepts

### HIP Runtime APIs

- `hipMalloc`: Allocates device memory
- `hipMemcpy`: Transfers data between host and device
- `hipFree`: Frees device memory
- `hipGetLastError`: Retrieves the last error from a runtime call
- `hipDeviceSynchronize`: Blocks until all device operations complete

### Atomic Operations

- `atomicAdd`: Atomically adds a value to a memory location
- Other atomic operations available in HIP:
  - `atomicSub`: Atomic subtraction
  - `atomicMax`: Atomic maximum
  - `atomicMin`: Atomic minimum
  - `atomicInc`: Atomic increment
  - `atomicDec`: Atomic decrement
  - `atomicCAS`: Compare and swap
  - `atomicExch`: Exchange

### Device Code Features

- `__global__`: Declares a kernel function callable from host
- `blockIdx`, `blockDim`, `threadIdx`: Built-in variables for grid/block indexing
- 2D thread indexing for image processing

### Histogram Algorithm

A histogram aggregates data points into discrete bins:

1. For each data point (pixel), determine its value (brightness)
2. Calculate which bin the value falls into
3. Increment the count for that bin
4. Repeat for all data points

The challenge: Step 3 requires atomic operations in parallel environments.

## Configuration

- Default input: `test.jpg`
- Number of bins: 256 (full 8-bit range)
- Block size: 16x16 threads
- Brightness calculation: Average of RGB channels
- Command line usage: `./hip_histogram_atomics [input.jpg]`

## Example Output

```bash
Loaded image: test.jpg
Dimensions: 1024 x 768
Channels: 3

Calculating histogram with 256 bins...
Launching kernel with grid size: (64, 48)
Block size: (16, 16)

Histogram (Brightness Distribution):
============================================================
Bin   0 [ 0.23%]: ### (1812)
Bin  16 [ 1.45%]: ############# (11402)
Bin  32 [ 2.87%]: ######################### (22595)
Bin  48 [ 3.21%]: ############################ (25267)
Bin  64 [ 4.56%]: ######################################## (35891)
...
============================================================

Statistics:
Total pixels: 786432
Mean brightness: 0.512
Number of bins: 256

Execution completed successfully.
```

## Understanding the Code

### Kernel Design

Each thread:

1. Computes its (x, y) coordinates from thread and block indices
2. Checks if coordinates are within image bounds
3. Calculates the pixel's index in the flattened array
4. Averages the RGB channels to get brightness
5. Maps brightness to a bin number
6. Uses `atomicAdd` to safely increment the bin

### Memory Layout

The image uses Channel-Height-Width layout:

- All red channel values, then green, then blue
- Index calculation: `(y * width + x) * channels + channel_offset`

### Why Atomic Operations Matter

Without atomics, the following race condition could occur:

```bash
Thread A reads bin[50] = 100
Thread B reads bin[50] = 100
Thread A writes bin[50] = 101
Thread B writes bin[50] = 101  // Lost Thread A's update!
```

With atomics, both increments are guaranteed to complete correctly, resulting in bin[50] = 102.

## Demonstrated API calls

### HIP runtime

#### Device symbols

- `atomicAdd`
- `blockDim`
- `blockIdx`
- `threadIdx`

#### Host symbols

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`

### External Libraries

- `stb_image.h`: Image loading (supports JPEG, PNG, BMP, TGA, etc.)
