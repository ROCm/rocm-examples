# Cooperative Groups Double-Buffered Tile Example

## Description

This program showcases a double-buffered tile load pipeline built from two cooperative groups
APIs: the group-collective `cooperative_groups::memcpy_async` and the split barrier of a
`thread_block`. A split barrier decomposes an ordinary block barrier into two phases,
`barrier_arrive` and `barrier_wait`, so that independent work can run between them instead of every
thread blocking immediately.

A single block streams a 1D array through two LDS (shared memory) buffers, one tile at a time.
The async load of the next tile is issued into the second buffer while the current tile is consumed.
A split barrier separates the moment a thread has finished reading a buffer from the moment the
block guarantees that every thread is done. The kernel applies the element-wise operation
`out[i] = scale * in[i] + bias`, which is trivial to validate against a CPU reference.

`cooperative_groups::memcpy_async` is an **asynchronous**, group-collective copy (typically
global <-> LDS). HIP does not expose a separate wait handle (there is no `cg::wait()`), so its
completion must be enforced by a following group barrier - either a `block.sync()` (as the official
reference test does) or, as in this example, the `barrier_wait` of a split barrier whose
`barrier_arrive` is issued *after* the copy. Ordering matters: the prefetch of the next tile is
issued **before** `barrier_arrive`, so the release fence in `barrier_arrive` orders the copy's
completion and the acquire fence in `barrier_wait` makes the prefetched buffer visible to every
thread by the next iteration. The split barrier additionally lets the current tile's computation
run as independent work between `barrier_arrive` and `barrier_wait`. Correctness (no data races,
validated output) is the top priority.

This example targets the AMD/HIP (ROCm) backend, and it requires a ROCm version recent enough to ship `hip/cooperative_groups/memcpy_async.h`.

### Application flow

1. A number of compile-time constants define the tile size and block size, the number of
   tiles, the total element count, and the constants of the element-wise operation.
2. The input array is set up in host memory and the output array is allocated.
3. The input is copied to the device.
4. The double-buffered pipeline kernel is launched in a single block.
   1. The first tile is loaded into the first LDS buffer with `memcpy_async`, followed by a
      block-wide `sync` that completes the async load and makes the tile visible to all threads.
   2. For each tile the block issues the async load of the next tile into the other buffer, calls
      `barrier_arrive`, then - as independent work between arrive and wait - consumes the current
      buffer (applies the element-wise operation and writes the result to global memory), and finally
      calls `barrier_wait` to complete the barrier and the in-flight prefetch.
5. The result array is copied back to the host and all device memory is freed.
6. The elements of the result are compared with the CPU reference. The result of the comparison is
   printed to the standard output.

## Key APIs and Concepts

- `cooperative_groups::this_thread_block` returns the `thread_block` group that represents all
  threads of the block. The block is used both as the group for the collective copies and as the
  group that owns the split barrier.
- `cooperative_groups::memcpy_async(group, dst, src, count_in_bytes)` is an asynchronous
  group-collective copy that is designed for global <-> LDS transfers. Every thread of the group
  must call it collectively. **In HIP the `count` argument is expressed in bytes** (here `tile_size *
  sizeof(float)`). The copy is asynchronous and HIP exposes no separate wait handle, so its
  completion is enforced by a following group barrier (`block.sync()` or
  `barrier_wait`). On hardware or compilers without the asynchronous LDS builtins
  `cooperative_groups::memcpy_async()` falls back to a
  traditional per-thread copy, so it always produces correct results.
- The split barrier decomposes a block barrier into two phases. `thread_block::barrier_arrive`
  signals that a thread has reached the barrier and returns an `arrival_token`; it emits a release
  fence and does not block, which exposes a window for independent work. `thread_block::barrier_wait`
  consumes the moved token, blocks until every thread of the block has arrived, and emits an acquire
  fence. Together they act as a full block barrier whose release/acquire fences order the prefetch
  issued just before `barrier_arrive`.
- Two LDS buffers (`__shared__ float buf[2][tile_size]`) are alternated between iterations so that
  the buffer being consumed is never the buffer being prefetched.
- Race-freedom: (a) the buffer prefetched during an iteration (`buf[(t + 1) & 1]`) is always
  different from the buffer read as independent work (`buf[cur]`), so the load and the reads target
  disjoint memory; (b) the prefetch is issued before `barrier_arrive`, so the barrier's
  release/acquire fences order its completion and visibility before the next iteration consumes it;
  (c) `buf[cur]` is only overwritten by the prefetch issued in the next iteration, which cannot begin
  until every thread has passed this iteration's `barrier_wait` - i.e. after every thread has
  finished reading `buf[cur]`. In this configuration the tile size equals the block size, so every
  thread also copies exactly the element it later reads.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `cooperative_groups::this_thread_block`
- `thread_block`
- `cooperative_groups::memcpy_async`
- `thread_block::barrier_arrive`
- `thread_block::barrier_wait`
- `thread_block::sync`
- All above from the [`cooperative_groups` namespace](https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_cooperative_groups.h)

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipStreamDefault`
- `hipGetLastError`
- `hipFree`
