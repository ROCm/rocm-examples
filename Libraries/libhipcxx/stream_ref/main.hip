// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "example_utils.hpp"

// <hip/stream_ref> provides hip::stream_ref, a small non-owning wrapper
// around a hipStream_t. In libhipcxx it is genuinely backed by hipStream_t
// (include/cuda/stream_ref, "using value_type = ::hipStream_t"), not a
// renamed CUDA type, so it drops into existing HIP code without a cast.
#include <hip/stream_ref>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

/// \brief Advances a simple linear congruential generator by one step. The
/// multiplier and increment are the constants used by Numerical Recipes; the
/// arithmetic wraps around modulo 2^32 using regular unsigned overflow.
__host__ __device__ constexpr uint32_t lcg_step(const uint32_t state) {
  return state * 1664525u + 1013904223u;
}

/// \brief Computes the buffer value for \p index by seeding the generator
/// with the index and advancing it \p iterations times. This function is
/// compiled for both host and device: the kernel calls it to fill the
/// buffer, and main() calls it again on the host to compute the reference
/// values used for validation, so both sides are guaranteed to run the exact
/// same arithmetic.
__host__ __device__ uint32_t busy_fill_value(const uint32_t index,
                                             const unsigned int iterations) {
  uint32_t state = index;
  for (unsigned int i = 0; i < iterations; ++i) {
    state = lcg_step(state);
  }
  return state;
}

/// \brief Fills \p out with busy_fill_value(index, iterations) for every
/// index. The loop inside busy_fill_value gives the kernel a modest, but
/// clearly non-zero, amount of real work to do, which is what makes the
/// hip::stream_ref::ready() check below meaningful to look at.
__global__ void busy_fill_kernel(uint32_t *out, const unsigned int count,
                                 const unsigned int iterations) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = busy_fill_value(index, iterations);
  }
}

/// \brief Prints the identity of the wrapped stream as its raw handle value.
///
/// \p ref is taken by value. hip::stream_ref's converting constructor from
/// hipStream_t (include/cuda/stream_ref, constructor at line 108) is
/// implicit, so this function can be called directly with a raw hipStream_t
/// handle: no explicit cast or wrapping is required at the call site. See the
/// call with `raw` in main() below.
void print_stream_identity(hip::stream_ref ref, const char *label) {
  std::cout << "  " << std::left << std::setw(32) << label
            << reinterpret_cast<void *>(ref.get()) << '\n';
}

/// \brief Calls hip::stream_ref::ready() for the racy, right-after-launch
/// poll and prints the outcome.
///
/// ready() (include/cuda/stream_ref, line 172) wraps hipStreamQuery: it
/// returns false for hipErrorNotReady and true for hipSuccess, but *throws*
/// hip::cuda_error for any other error. We deliberately catch that here
/// instead of letting it propagate out of main(): for this example we want
/// to keep running (print what happened, still validate the buffer) rather
/// than terminate on an unexpected runtime error, and a caught exception is
/// still fully reported via its .what() message, so nothing is silently
/// ignored. The returned value itself is not validated here: right after the
/// launch, false is merely the *expected* outcome, not a guaranteed one (see
/// the comment at the call site in main()), so the only thing worth counting
/// as an error is an unexpected exception.
void checked_ready(const hip::stream_ref &ref, const char *label, int &errors) {
  try {
    const bool ready = ref.ready();
    std::cout << "  " << std::left << std::setw(32) << label << std::boolalpha
              << ready << std::noboolalpha << '\n';
  } catch (const std::exception &e) {
    std::cerr << "  " << label << " threw: " << e.what() << '\n';
    ++errors;
  }
}

/// \brief Calls hip::stream_ref::ready() once wait() has returned and
/// validates the outcome.
///
/// Unlike checked_ready() above, this check is not racy: after wait() has
/// returned, every operation queued on the stream before it is guaranteed to
/// have completed, so ready() must report true. Both an exception and an
/// unexpected false are therefore counted as validation errors here, each
/// counted exactly once.
void checked_ready_after_wait(const hip::stream_ref &ref, int &errors) {
  try {
    const bool ready = ref.ready();
    std::cout << "  " << std::left << std::setw(32) << "ready() after wait()"
              << std::boolalpha << ready << std::noboolalpha << '\n';
    if (!ready) {
      ++errors;
    }
  } catch (const std::exception &e) {
    std::cerr << "  ready() after wait() threw: " << e.what() << '\n';
    ++errors;
  }
}

/// \brief Calls hip::stream_ref::wait() and reports whether it threw.
///
/// wait() (include/cuda/stream_ref, line 160) wraps hipStreamSynchronize and,
/// like ready(), throws hip::cuda_error on failure instead of the
/// HIP_CHECK-style "print and abort" used for the plain HIP calls elsewhere
/// in this file. See the comment on checked_ready() above for why this
/// example catches it rather than letting it propagate.
void checked_wait(const hip::stream_ref &ref, int &errors) {
  try {
    ref.wait();
  } catch (const std::exception &e) {
    std::cerr << "  hip::stream_ref::wait() threw: " << e.what() << '\n';
    ++errors;
  }
}

int main() {
  int errors = 0;

  // Number of elements in the buffer the kernel touches.
  constexpr unsigned int size = 1u << 16; // 65536

  // Number of threads per block.
  constexpr unsigned int block_size = 256;

  // Number of blocks, rounded up so that every element gets a thread.
  constexpr unsigned int grid_size = ceiling_div(size, block_size);

  // Number of LCG steps each thread performs. This is deliberately larger
  // than a single trivial write so that the kernel keeps the stream busy for
  // a modest, but non-zero, amount of time, which is what
  // hip::stream_ref::ready() below tries to observe. It is still small
  // enough that recomputing the same function on the host for validation
  // stays fast.
  constexpr unsigned int busy_iterations = 256;

  // Create an explicit HIP stream. hip::stream_ref never owns the stream it
  // wraps, so the hipStream_t below is what actually owns the resource and
  // must outlive every hip::stream_ref that refers to it.
  hipStream_t raw;
  HIP_CHECK(hipStreamCreate(&raw));

  // Wrap the raw handle. This constructor is implicit (see
  // print_stream_identity's doc comment above and its first call below), so
  // `s` behaves like a lightweight view of `raw`, not a separate resource.
  hip::stream_ref s{raw};

  // A default-constructed hip::stream_ref refers to the default stream
  // (its underlying handle is the null handle), which is a distinct stream
  // identity from the explicit stream created above. It does not create or
  // own anything, so no corresponding hipStreamCreate/hipStreamDestroy calls
  // are needed for it.
  const hip::stream_ref default_ref;

  std::cout << "Stream identities (raw handle values):\n";
  // Passing `raw` here relies on hip::stream_ref's implicit conversion from
  // hipStream_t: no `hip::stream_ref{raw}` wrapping is written at the call
  // site.
  print_stream_identity(raw, "explicit stream (raw)");
  print_stream_identity(s, "explicit stream (wrapped)");
  print_stream_identity(default_ref, "default stream");
  std::cout << '\n';

  // Allocate the device buffer that the kernel fills.
  uint32_t *d_result{};
  HIP_CHECK(hipMalloc(&d_result, sizeof(uint32_t) * size));

  // Launch the kernel on the explicit stream. The stream_ref itself has no
  // implicit conversion back to hipStream_t (only the other direction is
  // implicit), so the raw handle must be requested explicitly via get() to
  // pass it to the <<<...>>> launch syntax, which always expects a
  // hipStream_t.
  busy_fill_kernel<<<grid_size, block_size, 0, s.get()>>>(d_result, size,
                                                          busy_iterations);
  HIP_CHECK(hipGetLastError());

  std::cout << "Polling and joining the stream:\n";

  // Immediately after the launch, poll the stream instead of blocking on it.
  // The kernel above does a modest, but non-trivial, amount of work per
  // thread, so at this point the stream is expected to still be busy and
  // ready() is expected to report false. This check is inherently racy: it
  // is a best-effort snapshot of the stream's state at the moment
  // hipStreamQuery runs, not a guarantee, and depends on how fast the GPU
  // finishes relative to how fast the host reaches this line. It is printed
  // for illustration only and is not asserted.
  checked_ready(s, "ready() right after launch", errors);

  // Block until the kernel completes.
  checked_wait(s, errors);

  // Once wait() has returned, the stream is guaranteed to have completed
  // every operation queued before it, so this second ready() check is not
  // racy: it must report true, and it is treated as a validation error if it
  // does not.
  checked_ready_after_wait(s, errors);

  // get() hands the raw handle back out for a HIP API that hip::stream_ref
  // does not wrap itself. Here it is used for a second, redundant explicit
  // join: calling hipStreamSynchronize again on an already-idle stream is
  // well defined and simply returns immediately.
  HIP_CHECK(hipStreamSynchronize(s.get()));
  std::cout << '\n';

  // Copy the results back to the host.
  std::vector<uint32_t> h_result(size);
  HIP_CHECK(hipMemcpy(h_result.data(), d_result, sizeof(uint32_t) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_result));

  // Print a few values so the fill pattern can be checked by eye.
  constexpr unsigned int print_count = 8;
  std::cout << "First " << print_count << " values of the filled buffer:\n"
            << "index value\n";
  for (unsigned int i = 0; i < print_count; ++i) {
    std::cout << std::setw(5) << i << ' ' << std::setw(10) << h_result[i]
              << '\n';
  }
  std::cout << '\n';

  // Demonstrate identity/equality. hip::stream_ref compares by the wrapped
  // handle's value, so two stream_refs that wrap the same hipStream_t
  // compare equal, and the comparison also works directly against a raw
  // hipStream_t thanks to the implicit conversion.
  const hip::stream_ref s_alias{raw};
  const bool same_stream_equal = (s == s_alias);
  const bool raw_conversion_equal = (s == raw);
  const bool default_stream_distinct = (s != default_ref);

  std::cout << "Identity and equality checks:\n"
            << "  s == s_alias (same raw stream):        " << std::boolalpha
            << same_stream_equal << '\n'
            << "  s == raw (compare against hipStream_t): "
            << raw_conversion_equal << '\n'
            << "  s != default_ref (distinct identity):   "
            << default_stream_distinct << std::noboolalpha << '\n'
            << std::endl;

  if (!same_stream_equal) {
    ++errors;
  }
  if (!raw_conversion_equal) {
    ++errors;
  }
  if (!default_stream_distinct) {
    ++errors;
  }

  // Validate the filled buffer against the host-computed reference. Both
  // sides call the exact same busy_fill_value() function, so any mismatch
  // indicates a real problem with the kernel launch or the copy, not a
  // difference between two independently written implementations.
  for (unsigned int i = 0; i < size; ++i) {
    if (h_result[i] != busy_fill_value(i, busy_iterations)) {
      ++errors;
    }
  }

  // Destroying the stream is another HIP API that hip::stream_ref does not
  // wrap: it is a non-owning view, so it has no destructor logic of its own.
  // get() hands back the same raw handle that was used to construct `s` and
  // `s_alias` for the one HIP call that actually releases the resource.
  HIP_CHECK(hipStreamDestroy(s.get()));

  return report_validation_result(errors);
}
