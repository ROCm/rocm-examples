# libhipcxx Atomic Counter Example

## Description

This example shows how libhipcxx's atomic types relate to each other by using every one of them to build the same thing: a counter that many threads increment concurrently. It combines three groups of libhipcxx atomic types, in the order they are introduced below:

- `cuda::std::atomic<int>`, the conforming C++ Standard atomic, usable in both `__host__` and `__device__` code.
- `cuda::atomic<int, cuda::thread_scope_device>`, the libhipcxx extension that adds an explicit thread scope to the same atomic.
- `cuda::std::atomic_ref<int>` and `cuda::atomic_ref<int, cuda::thread_scope_device>`, which layer atomic access over a plain, non-atomic `int` instead of owning their own storage.

### Application flow

1. Four device counters are allocated: two `cuda::std::atomic<int>` / `cuda::atomic<int, cuda::thread_scope_device>` objects, and two plain `int`s that will only ever be accessed through an `atomic_ref`. All four are zero-initialized with `hipMemset`.
2. Four kernels are launched, one per counter, each with one thread per increment:
   1. `increment_std_atomic_kernel` calls `fetch_add(1, cuda::std::memory_order_relaxed)` on a `cuda::std::atomic<int> *`.
   2. `increment_device_atomic_kernel` does the same on a `cuda::atomic<int, cuda::thread_scope_device> *`, using `cuda::memory_order_relaxed`.
   3. `increment_std_atomic_ref_kernel` wraps a plain `int *` in a `cuda::std::atomic_ref<int>` and calls `fetch_add(1, cuda::std::memory_order_relaxed)` on that.
   4. `increment_device_atomic_ref_kernel` does the same with `cuda::atomic_ref<int, cuda::thread_scope_device>` and `cuda::memory_order_relaxed`.
3. The four results are copied back to the host as plain `int`s and the device memory is freed.
4. The four results are printed to the standard output so they can be checked by eye.
5. The host validates that every counter equals the number of threads that incremented it: if any increment were lost to a data race, the corresponding counter would be smaller.

## Key APIs and Concepts

- `cuda::std::atomic<T>` is the conforming C++ Standard atomic. Its spelling is identical to host-only `std::atomic<T>`; only the include changes, from `<atomic>` to `<cuda/atomic>`. Unlike `std::atomic`, it can be constructed, loaded, stored, and modified from both `__host__` and `__device__` code.
- `cuda::atomic<T, Scope>` is a libhipcxx extension of `cuda::std::atomic` with a second, optional template parameter: a `cuda::thread_scope`, the set of threads whose accesses to the atomic are guaranteed to synchronize with each other. It defaults to `cuda::thread_scope_system`, the same scope `cuda::std::atomic` is implicitly locked to, so `cuda::atomic<int>` without a second argument and `cuda::std::atomic<int>` are the same type in every way that matters: same layout, same operations, same behavior. The extension is the *ability to narrow* the scope, not a different default. Narrowing to `cuda::thread_scope_device` (as this example does) or `cuda::thread_scope_block` tells the compiler that only device threads (or only threads in the same block) will ever touch the atomic, which can avoid synchronization that a wider scope would require.
- `cuda::std::atomic_ref<T>` and its scoped counterpart `cuda::atomic_ref<T, Scope>` do not own the value they protect. They are constructed from a reference to a plain, ordinary object (here, `int`) and make only the accesses that go *through* the reference atomic; any other access to the same object that bypasses every live `atomic_ref` referring to it is still a data race. This is useful when the storage already exists in a shape you cannot change (for example, a member of a larger struct) but still needs atomic access.
- `cuda::std::atomic<int>` and `cuda::atomic<int, Scope>` store nothing but a single `int`: the underlying storage is a plain, trivially copyable value, so `hipMemset` can zero-initialize it in place and a raw `hipMemcpy` into a plain host `int` is enough to read it back. No atomic object is ever constructed on the host in this example.
- `fetch_add(1, order)` atomically adds `1` to the current value and returns the value from *before* the addition; every atomic type shown here provides it with the same signature. Every call below passes `memory_order_relaxed` (spelled `cuda::std::memory_order_relaxed` for the `cuda::std::` types and `cuda::memory_order_relaxed`, the same enumerator, for the `cuda::` types): a pure counter only needs the increments to be atomic, not ordered with respect to other memory operations, and the final value is only read after `hipMemcpy` has synchronized with the device.

## Demonstrated API Calls

### libhipcxx

- `cuda::std::atomic`
- `cuda::atomic`
- `cuda::std::atomic_ref`
- `cuda::atomic_ref`
- `cuda::thread_scope_device`
- `cuda::std::memory_order_relaxed`
- `cuda::memory_order_relaxed`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `hipMalloc`
- `hipMemset`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`

> `hip::atomic` and `<hip/atomic>` are the same entities as `cuda::atomic` and `<cuda/atomic>`: `hip` is a namespace alias for `cuda` throughout libhipcxx, provided for readers who prefer to spell the HIP port's types without the `cuda::` prefix.
