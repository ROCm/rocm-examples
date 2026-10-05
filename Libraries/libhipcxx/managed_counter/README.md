# libhipcxx Managed Memory Atomic Counter Example

## Description

This example shows how to increment one counter from the host and the device *at the same time* without losing an update. The counter lives in managed memory, allocated with `hipMallocManaged`, so the host and the device share the same memory and no `hipMemcpy` is needed to hand data between them. The host threads and the device threads then race on the same `int`: the atomics guarantee that every `fetch_add` is atomic, so no increment can be lost.

The counter itself stays a plain `int` for the whole example. All atomicity comes from `hip::std::atomic_ref`, which the host threads and the device threads construct over the shared memory.

> Managed memory requires a platform that supports it. The [unified memory documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/unified_memory.html) lists the supported operating systems and GPUs.

### Application flow

1. One `int` of managed memory is allocated with `hipMallocManaged` and zero-initialized in place. `hipMallocManaged` does not run a constructor, so the example writes a `0` into the `int` with placement new.
2. A host thread is started. It wraps the shared `int` in its own `hip::std::atomic_ref<int>` and applies `fetch_add(1, hip::std::memory_order_relaxed)` 10,000 times.
3. While the host thread runs, the `increment_kernel` kernel is launched with one thread per increment (10,000 threads, rounded up to whole blocks), and a second host thread is started. Every device thread and both host threads wrap the same `int` in their own `hip::std::atomic_ref<int>` and apply `fetch_add(1, hip::std::memory_order_relaxed)`, so host and device increments interleave freely.
4. The host threads are joined and the device is synchronized with `hipDeviceSynchronize`. Both sides are now finished.
5. The final counter value is printed, then validated: it must equal the number of host increments plus the number of device increments. A smaller value would mean an increment was lost to a data race, which the atomics forbid.

## Key APIs and Concepts

- `hip::std::atomic_ref<T>` wraps a plain, ordinary `T` and makes the accesses that go *through* it atomic. It does not own the value it protects, so the shared counter can stay a plain `int` that the host and the device both address. Any access to the counter that does not go through a live `atomic_ref` while other threads increment it would still be a data race.
- `hipMallocManaged` allocates memory that is addressable from the host and the device. `hipFree` releases it.
- Each thread constructs its own `atomic_ref` over the same `int`. Copies of an `atomic_ref` all refer to the same object, so every `fetch_add` through every copy is atomic with respect to every other copy's.
- `fetch_add(1, order)` atomically adds `1` to the current value and returns the value from *before* the addition. `memory_order_relaxed` is sufficient: a pure counter needs the increments to be atomic, not ordered with respect to other memory operations, and both sides finish before the final value is read.
- `std::thread` and `hipDeviceSynchronize` provide the join points: the host threads are joined with `thread.join`, the device work is awaited with `hipDeviceSynchronize`, and only then is the final value read.

## Demonstrated API Calls

### libhipcxx

- `hip::std::atomic_ref`
- `hip::std::memory_order_relaxed`
- `fetch_add`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `hipMallocManaged`
- `hipGetLastError`
- `hipDeviceSynchronize`
- `hipFree`
- `std::thread` (host-side)
