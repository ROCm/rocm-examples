# libhipcxx Stream Ref Example

## Description

This example shows how to use `cuda::stream_ref`, a small, non-owning wrapper around a `hipStream_t`. Unlike some other libhipcxx types, `cuda::stream_ref` is genuinely backed by `hipStream_t` on AMD GPUs, not a renamed CUDA type, so it can be used as a drop-in wrapper around existing HIP streams. Its converting constructor from `hipStream_t` is implicit, so a function that takes a `cuda::stream_ref` by value can be called directly with a raw `hipStream_t` handle, with no explicit cast or wrapping needed.

The example creates an explicit HIP stream, wraps it, launches a kernel that does a modest, deliberately non-trivial amount of work on that stream, and then demonstrates the operations `cuda::stream_ref` offers around it: a non-blocking poll with `ready()`, a blocking join with `wait()`, retrieving the raw handle with `get()` for a HIP API the wrapper does not cover, and identity/equality comparisons.

### Application flow

1. An explicit HIP stream is created with `hipStreamCreate` and wrapped in a `cuda::stream_ref`.
2. A default-constructed `cuda::stream_ref` is created, which refers to the default stream. It is a distinct stream identity from the explicit stream, even though it does not own or create anything itself.
3. The raw handle values behind the explicit stream, the wrapped `cuda::stream_ref`, and the default stream are printed so that their identities can be compared by eye. These printed handles are addresses and differ between runs (and between processes); only their equality or inequality to each other is meaningful, not their specific values.
4. Device memory is allocated for the output buffer.
5. A kernel is launched on the explicit stream, obtained via `s.get()`. Each thread seeds a small linear congruential generator with its index and advances it a fixed number of times, giving the kernel a modest, non-zero amount of real work to do.
6. Immediately after the launch, `cuda::stream_ref::ready()` is called and the result is printed. At this point the stream is expected to still be busy, so `ready()` is expected to report `false`, but this check is inherently racy and best-effort: depending on how fast the GPU finishes relative to the host, it may print either `true` or `false`. It is shown for illustration only and not asserted.
7. `cuda::stream_ref::wait()` blocks until the kernel has completed. `ready()` is called again afterward; this time it is not racy, and must report `true`.
8. `s.get()` is used to call `hipStreamSynchronize` directly, a second, redundant explicit join that demonstrates handing the raw handle to a HIP API that `cuda::stream_ref` does not wrap itself.
9. The results are copied back to the host, and a few values are printed so the fill pattern can be checked by eye.
10. Identity and equality are demonstrated: a second `cuda::stream_ref` wrapping the same raw stream compares equal to the first, a `cuda::stream_ref` compares equal directly against the raw `hipStream_t` it wraps (via the implicit conversion), and the default stream's `cuda::stream_ref` compares unequal to the explicit stream's.
11. The buffer is validated against a host-computed reference, computed by calling the exact same host/device function that the kernel uses, and the result of the comparison is printed to the standard output.
12. `s.get()` is used once more, this time to call `hipStreamDestroy`: destroying the stream is exactly the kind of raw-API operation that a non-owning wrapper does not cover.

## Key APIs and Concepts

- `cuda::stream_ref` is a small, non-owning wrapper around a `hipStream_t`. It is header-only and host-side; it never creates or destroys a stream itself.
- `cuda::stream_ref(hipStream_t)` is an implicit converting constructor, so functions that take a `cuda::stream_ref` parameter by value accept a raw `hipStream_t` argument directly.
- A default-constructed `cuda::stream_ref` refers to the default stream. It is a distinct identity from any explicitly created stream.
- `cuda::stream_ref::get()` returns the wrapped `hipStream_t` handle, for use with HIP APIs the wrapper does not cover, such as kernel launches, `hipStreamSynchronize`, or `hipStreamDestroy`.
- `cuda::stream_ref::ready()` wraps `hipStreamQuery`: it returns `false` if the stream has unfinished work (`hipErrorNotReady`), and `true` if all queued operations have completed. Any other error is reported by throwing `hip::cuda_error`.
- `cuda::stream_ref::wait()` wraps `hipStreamSynchronize` and blocks until the stream is idle. It also throws `hip::cuda_error` on failure, unlike the `HIP_CHECK`-style "print and abort" used for the plain HIP calls in this example.
- `hip::cuda_error` is an alias for `cuda::cuda_error`, which derives from `std::runtime_error` (and so from `std::exception`); its `what()` message includes the numeric HIP error code and a short description.
- Two `cuda::stream_ref` values compare equal if they wrap the same underlying `hipStream_t` handle. The comparison operators also accept a raw `hipStream_t` directly, again thanks to the implicit conversion.

## Demonstrated API Calls

### libhipcxx

- `cuda::stream_ref` (constructor, including the implicit conversion from `hipStream_t`)
- `cuda::stream_ref::get`
- `cuda::stream_ref::ready`
- `cuda::stream_ref::wait`
- `cuda::stream_ref::operator==`
- `cuda::stream_ref::operator!=`

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `hipStreamCreate`
- `hipStreamSynchronize`
- `hipStreamDestroy`
- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`
