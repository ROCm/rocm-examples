# libhipcxx Thread Scopes Example

## Description

This example shows why you would narrow a `hip::atomic`'s thread scope, rather than always using the widest one. `hip::thread_scope` selects which set of threads an atomic operation must synchronize with:

- `hip::thread_scope_thread` — the calling thread only.
- `hip::thread_scope_block` — threads of the same HIP block.
- `hip::thread_scope_device` — threads anywhere on the same GPU.
- `hip::thread_scope_system` — also threads on the host, or on other GPUs.

Scope selection is both a correctness decision (which threads must observe the update) and a performance decision (how far the update has to travel), and it has no analogue in host-only C++, where `std::atomic` is always visible to the whole process. The example builds one staged reduction that uses two of these scopes for two different purposes:

1. A block-scope atomic, placed in LDS (shared memory), combines the values contributed by the threads of one block. `hip::thread_scope_block` is sufficient because no other block ever touches this atomic, and confining the read-modify-write traffic to LDS makes it far cheaper than a device-scope atomic would be.
2. A device-scope atomic, placed in global memory, combines the per-block results into the final answer. `hip::thread_scope_device` is required here because every block in the grid updates the same atomic.

Both stages use `fetch_max`, a non-standard atomic operation that libhipcxx adds to `hip::atomic` and `hip::atomic_ref` (it does not exist on `hip::std::atomic`).

Note: the `hip::thread_scope` enum defines exactly `thread_scope_system`, `thread_scope_device`, `thread_scope_block` and `thread_scope_thread` — there is no `thread_scope_cluster`, even though HIP itself supports thread block clusters on some GPUs, for example gfx1250.

### Application flow

1. Device memory is allocated for the device-scope atomic, for one partial maximum per block, and for the final result.
2. `init_device_max_kernel` runs with a single thread and stores the identity element for max (`INT_MIN`) into the device-scope atomic. Because HIP orders kernel launches on the same stream, this completes before any other kernel touches the atomic.
3. `block_reduce_max_kernel` is launched with one thread per value. Each thread:
   1. Derives a value from its block and thread index with `thread_value`, chosen so the value varies visibly both within a block and across blocks.
   2. Initializes the block's `__shared__` block-scope atomic to `INT_MIN` (thread 0 only), then every thread in the block synchronizes with `__syncthreads()`.
   3. Applies `fetch_max` on the block-scope atomic with its own value, then synchronizes again so every thread's update has completed.
   4. Has thread 0 read the block's maximum, write it to the block's slot in device memory, and apply `fetch_max` on the device-scope atomic with it.
4. `load_device_max_kernel` runs with a single thread and copies the device-scope atomic's value into a plain `int` in device memory, so the host does not have to assume anything about the atomic's internal storage layout.
5. The per-block maxima and the final result are copied back to the host, and the device memory is freed.
6. The host independently computes the same maxima with `thread_value`, prints both the per-block maxima and the final device-scope maximum next to their reference values, and validates that they match.

## Key APIs and Concepts

- `hip::atomic<T, Scope>` is a libhipcxx atomic that carries its thread scope as a template argument, unlike `hip::std::atomic<T>` (which is always `thread_scope_system`).
- `hip::thread_scope_block` and `hip::thread_scope_device` select the set of threads an atomic operation is guaranteed to synchronize with. Choosing the narrowest scope that is still correct reduces how far a read-modify-write has to travel: a block-scope atomic that lives in LDS never leaves the compute unit that owns the block, while a device-scope atomic must be visible to every block in the grid.
- `fetch_max` (and its counterpart `fetch_min`) are non-standard atomic operations added by libhipcxx to `hip::atomic` and `hip::atomic_ref`. They are not available on `hip::std::atomic`.
- A `__shared__` variable never runs a constructor, and shared memory is not zero-initialized between kernel launches. `hip::atomic`'s default constructor is defaulted, and every implementation layer behind it only holds a trivially-constructible storage member, so the type as a whole is trivial: its object lifetime begins as soon as the variable's storage exists, and calling `store()` on it without placement-new is well defined. What still must happen explicitly is giving it a known *value* before it is used: exactly one thread stores the identity element, and every thread synchronizes with `__syncthreads()` before relying on it.
- The device-scope atomic is initialized the same way, but from a separate single-thread kernel launched before the main kernel, because HIP guarantees kernels launched on the same stream execute in order; no explicit device-wide barrier is required between them.

## Demonstrated API Calls

### libhipcxx

- `hip::atomic`
- `hip::thread_scope_block`
- `hip::thread_scope_device`
- `hip::memory_order_relaxed`
- `fetch_max`
- `store`
- `load`

### HIP runtime

#### Device symbols

- `blockIdx`
- `threadIdx`
- `__shared__`
- `__syncthreads`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
