# Cooperative Groups Prefix Sum Example

## Description

This program showcases a parallel prefix sum (scan) computed with the Cooperative Groups scan API.

The block is partitioned into warp-sized `thread_block_tile`s, and each tile independently computes both an *inclusive* and an *exclusive* prefix sum over its own contiguous segment of the input array:

- The inclusive scan returns, for every element, the sum of all elements up to and including itself.
- The exclusive scan returns, for every element, the sum of all elements strictly before itself.

Both scans use the `plus` reduction operator, which is the default for `inclusive_scan` and `exclusive_scan`.

The tile is instantiated at the device warp size (64 on CDNA GPUs such as gfx9xx/gfx94x, 32 on RDNA GPUs such as gfx10/gfx11). When the tile size equals the warp size and the element type is primitive, the scan takes advantage of the DPP (Data-Parallel Primitives) hardware acceleration path.

### Application flow

1. The warp size of the current device is queried via `hipGetDeviceProperties`.
2. Because the `tiled_partition` size must be known at compile time, the program dispatches to a kernel instantiated at the device warp size (32 or 64). If the device reports any other warp size, a message is printed and the program exits gracefully.
3. A number of variables are defined to control the problem details and the kernel launch parameters. The block size is chosen as a small multiple of the tile size so that the array is partitioned into independent, warp-sized segments.
4. The input array is set up in host memory with a deterministic pattern.
5. The input is copied to the device.
6. The scan kernel is launched with the previously defined arguments. Each warp-sized tile loads one element, then performs an inclusive scan and an exclusive scan over its segment, writing both results to global memory.
7. The result arrays are copied back to the host and all device memory is freed.
8. A CPU reference computes the per-tile inclusive and exclusive prefix sums, and the elements of the result arrays are compared with the expected results. The result of the comparison is printed to the standard output.

## Key APIs and Concepts

- `cooperative_groups::this_thread_block()` returns the `thread_block` group consisting of all threads in the block.
- `cooperative_groups::tiled_partition<Size>(group)` partitions a group into a static `thread_block_tile<Size>`. Instantiating the tile at the device warp size is what enables the DPP-accelerated scan path for primitive element types.
- `cooperative_groups::inclusive_scan(tile, val)` and `cooperative_groups::exclusive_scan(tile, val)` compute prefix sums across the lanes of the tile. They use `cooperative_groups::plus` by default; an explicit binary operator can also be supplied.
- The scans are *collective* operations: every lane of the tile must participate. Out-of-bounds lanes therefore contribute the additive identity (0) rather than returning early.
- The first-lane (lane 0) return value of `exclusive_scan` is platform-dependent in general for arbitrary operators. However, for `plus` the identity is 0 on both AMD and NVIDIA, so lane 0 returns 0, which matches the mathematically correct exclusive prefix. Validation of the `plus` exclusive scan therefore works on both platforms without any special-casing.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `cooperative_groups::this_thread_block`
- `cooperative_groups::tiled_partition`
- `thread_block`
- `thread_block_tile`
- `cooperative_groups::inclusive_scan`
- `cooperative_groups::exclusive_scan`
- `cooperative_groups::plus`

#### Host symbols

- `hipGetDeviceProperties`
- `hipMalloc`
- `hipMemcpy`
- `hipLaunchKernelGGL`
- `hipGetLastError`
- `hipFree`
