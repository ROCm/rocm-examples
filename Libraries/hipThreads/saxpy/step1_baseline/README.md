# hipThreads SAXPY Step 1: CPU Baseline Example

## Description

This example demonstrates a compute-bound SAXPY (Single-precision A·X + Y) implementation on the CPU using `std::thread`. Each element undergoes 512 iterations of the expression `t = a*t + y[i]`, making the workload deliberately compute-bound rather than memory-bound.

This is the starting point of the SAXPY porting series. Steps 2 and 3 progressively port this CPU code to the GPU using hipThreads.

### Application flow

1. Allocate and initialize two host vectors `x` (all 1.0) and `y` (all 2.0) of 268 million elements each.
2. Spawn `std::thread::hardware_concurrency()` threads, partitioning the element range evenly across them.
3. Each thread runs the compute-bound saxpy kernel: 512 iterations of `t = a*t + y[i]` per element, writing the result back to `y`.
4. Join all threads and measure elapsed time.
5. Validate the output against a CPU reference value and print the elapsed time in nanoseconds.
