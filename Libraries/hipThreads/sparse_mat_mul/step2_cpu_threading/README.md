# hipThreads Sparse Matrix Multiplication Step 2: CPU Threading Example

## Description

This example extends the step 1 single-threaded SpMM baseline with multi-threaded parallelism on the CPU using `std::thread`. The multiplication loop over rows of A is partitioned across threads, with each thread processing a contiguous chunk of rows. A parallel row sort is also added to the CSR construction step to reduce preprocessing time.

### Application flow

1. Read and convert matrix A to CSR format, using parallel row sorting.
2. Read and convert matrix B to CSC format.
3. Partition the rows of A evenly across `std::thread::hardware_concurrency()` threads.
4. Each thread computes its assigned rows of C = A × B independently.
5. Join all threads and print the result summary and elapsed time.
