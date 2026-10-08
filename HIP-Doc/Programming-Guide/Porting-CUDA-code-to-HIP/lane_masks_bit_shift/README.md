# HIP-Doc Lane Masks Bit-Shift Example

## Description

This example demonstrates how to write portable HIP code that correctly handles
lane masks and bit-shift operations across different GPU architectures with
varying warp sizes. It highlights a common portability issue when porting CUDA
code to HIP and provides best-practice solutions.

For more information on this topic, please refer to the
[HIP Porting Guide - Lane masks bit-shift section](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#lane-masks-bit-shift).

### Background

A thread in a warp is also called a lane, and a lane mask is a bitmask where
each bit corresponds to a thread in a warp. Lane masks are commonly used in
warp-level operations for tasks like warp reduction, ballot operations, and
parallel algorithms.

The critical portability issue arises from the fact that:

- **AMD RDNA architectures** and **NVIDIA GPUs** have a warp size of 32
- **AMD GCN/CDNA architectures** have a warp size of 64

When using bit-shift operations to create lane masks, using 32-bit integer
literals (like `1 << laneId`) will overflow on AMD GPUs with a warp size of 64
and `laneId >= 32`, leading to incorrect results.

### Application Flow

The example demonstrates three approaches to handling lane masks:

1. **Problematic Implementation** (32-bit, non-portable)
   - Uses 32-bit integer literals for bit-shift operations
   - Only safe for warp sizes ≤ 32
   - Demonstrates the problem that needs to be avoided

2. **Improved Implementation** (64-bit, explicit)
   - Uses 64-bit integer literals (`1ull`) for bit-shift operations
   - Works correctly for all warp sizes
   - Explicit but not architecture-specific

3. **Best Practice Implementation** (portable)
   - Uses a portable `lane_mask_t` typedef that adapts to the architecture
   - Provides architecture-specific optimizations
   - Recommended approach for production code

For each approach, the example:

- Queries the device's warp size
- Executes kernels performing warp reduction with bit-shift operations
- Stores and displays the results
- Verifies correctness by comparing results

## Key APIs and Concepts

- `warpSize` - Built-in constant that represents the warp size in device code
- `hipDeviceGetAttribute` - Queries the device's warp size on the host side
- **Lane masks** - Bitmasks representing active threads in a warp
- **Bit-shift operations** - Creating masks and selecting threads using bit
  shifting
- **Architecture-specific typedefs** - Using conditional compilation for
  portability
- `hipcub::WarpReduce` - Warp-level reduction primitive from the hipCUB library

### Problem Demonstration

```cpp
// PROBLEMATIC - Only works on warp size ≤ 32
unsigned int laneId = threadIdx.x % warpSize;
std::uint32_t mask = (1 << laneId) - 1;  // Overflow when laneId >= 32!
```

### Solution 1: Explicit 64-bit

```cpp
// IMPROVED - Works on all current GPUs
unsigned int laneId = threadIdx.x % warpSize;
std::uint64_t mask = (1ull << laneId) - 1;  // Uses 64-bit literal
```

### Solution 2: Portable Typedef (Recommended)

```cpp
// BEST PRACTICE - Architecture-specific optimization
#if defined(__GFX8__) || defined(__GFX9__)
typedef std::uint64_t lane_mask_t;
#else
typedef std::uint32_t lane_mask_t;
#endif

unsigned int laneId = threadIdx.x % warpSize;
lane_mask_t mask = (lane_mask_t{1} << laneId) - 1;  // Portable!
```

## Building

### Linux

#### CMake

```bash
mkdir build && cd build
cmake ..  # or cmake -D GPU_RUNTIME=CUDA .. for CUDA
make
```

#### Make

```bash
make  # or make GPU_RUNTIME=CUDA for CUDA
```

### Windows

#### Visual Studio

Open one of the Visual Studio solution files and build the project:

- `lane_masks_bit_shift_vs2017.sln` for Visual Studio 2017
- `lane_masks_bit_shift_vs2019.sln` for Visual Studio 2019
- `lane_masks_bit_shift_vs2022.sln` for Visual Studio 2022

#### CMake

```shell
cmake -G Ninja -S . -B build
cmake --build build
```

## Running

After building, run the executable:

```shell
./hip_lane_masks_bit_shift  # Linux
hip_lane_masks_bit_shift.exe  # Windows
```

### Expected Output

On a GPU with warp size 64:

```plaintext
=== Device Information ===
Device name: AMD Instinct GPU
Warp size: 64

=== Test 1: Skipped ===
Problematic 32-bit kernel skipped (warpSize=64 would cause overflow)

=== Test 2: Improved 64-bit Implementation ===
Results: 9223372036854775808 9223372036854775808

=== Test 3: Best Practice Portable Implementation ===
Using lane_mask_t (64-bit on this architecture)
Results: 9223372036854775808 9223372036854775808

=== Verification ===
64-bit and portable results match: YES

SUCCESS: All kernels produced correct results!
```

On a GPU with warp size 32:

```plaintext
=== Device Information ===
Device name: AMD Radeon GPU
Warp size: 32

=== Test 1: Problematic 32-bit Implementation ===
Running on warpSize=32 (safe for this architecture)
Results: 2147483648 2147483648 2147483648 2147483648

=== Test 2: Improved 64-bit Implementation ===
Results: 2147483648 2147483648 2147483648 2147483648

=== Test 3: Best Practice Portable Implementation ===
Using lane_mask_t (32-bit on this architecture)
Results: 2147483648 2147483648 2147483648 2147483648

=== Verification ===
64-bit and portable results match: YES

SUCCESS: All kernels produced correct results!
```

## Demonstrated API Calls

### HIP Runtime

#### Device Symbols

- `threadIdx` - Thread index within a block
- `warpSize` - Built-in constant for warp size in device code

#### Host Symbols

- `hipDeviceGetAttribute` - Query device attributes
- `hipGetDeviceProperties` - Get device properties
- `hipMalloc` - Allocate device memory
- `hipMemcpy` - Copy memory between host and device
- `hipFree` - Free device memory
- `hipDeviceSynchronize` - Synchronize device execution
- `hipGetLastError` - Get last error from runtime

### hipCUB

- `hipcub::WarpReduce` - Warp-level reduction operations
