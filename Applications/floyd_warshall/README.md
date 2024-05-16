# Applications Floyd-Warshall Example

## Description

This example showcases a GPU implementation of the [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm), which computes the shortest path between each pair of nodes in a given directed and (in this case) complete graph $G = (V, E, \omega)$. The key point of this implementation is that each kernel launch represents a step $k$ of the traditional CPU-implemented algorithm. Therefore, the kernel is launched as much times as nodes $\left(n = \vert V \vert \right)$ has the graph.

In this example, there are `iterations` (consecutive) executions of the algorithm on the same graph. As each execution requires an unmodified graph input, multiple copy operations are required. Hence, the performance of the example can be improved by using _pinned memory_.

Pinned memory is simply a special kind of memory that cannot be paged out the physical memory of a process, meaning that the virtual addresses associated with it are always mapped to physical memory. When copying data from/to the host to/from the GPU, if host source/destination is not pinned memory the runtime and the operating system has to do ensure that the memory is not swapped out. This usually significantly impact the performance of memory movements.

Therefore, using pinned memory saves significant time needed to copy from/to host memory. In this example, performances is improved by using this type of memory, given that there are `iterations` (consecutive) executions of the algorithm on the same graph.

### Application flow

1. Default values for the number of nodes of the graph and the number of iterations for the algorithm execution are set.
2. Command line arguments are parsed (if any) and the previous values are updated.
3. A number of constants are defined for kernel execution and input/output data size.
4. Host memory is allocated for the distance matrix and initialized with the increasing sequence $1,2,3,\dots$ . These values represent the weights of the edges of the graph.
5. Host memory is allocated for the adjacency matrix and initialized such that the initial path between each pair of vertices $x,y \in V$ ($x \neq y$) is the edge $(x,y)$.
6. Pinned host memory and device memory are allocated. Data is first copied to the pinned host memory and then to the device. Memory is initialized with the input matrices (distance and adjacency) representing the graph $G$ and the Floyd-Warshall kernel is executed for each node of the graph.
7. The resulting distance and adjacency matrices are copied to the host and pinned memory and device memory are freed.
8. The mean time in milliseconds needed for each iteration is printed to standard output.
9. The results obtained are compared with the CPU implementation of the algorithm. The result of the comparison is printed to the standard output.

### Command line interface

There are three parameters available:

- `-h` displays information about the available parameters and their default values.
- `-n nodes` sets `nodes` as the number of nodes of the graph to which the Floyd-Warshall algorithm will be applied. It must be a (positive) multiple of `block_size` (= 16). Its default value is 16.
- `-i iterations` sets `iterations` as the number of times that the algorithm will be applied to the (same) graph. It must be an integer greater than 0. Its default value is 1.

## Key APIs and Concepts

- For this GPU implementation of the Floyd-Warshall algorithm, the main kernel (`floyd_warshall_kernel`) that is launched in a 2-dimensional grid. Each thread in the grid computes the shortest path between two nodes of the graph at a certain step $k$ $\left(0 \leq k < n \right)$. The threads compare the previously computed shortest paths using only the nodes in $V'=\{v_0,v_1,...,v_{k-1}\} \subseteq V$ as intermediate nodes with the paths that include node $v_k$ as an intermediate node, and take the shortest option. Therefore, the kernel is launched $n$ times.

- For improved performance, pinned memory is used to pass the results obtained in each iteration to the next one. With `hipHostMalloc` pinned host memory (accessible by the device) can be allocated, and `hipHostFree` frees it. In this example, host pinned memory is allocated using the `hipHostMallocMapped` flag, which indicates that `hipHostMalloc` must map the allocation into the address space of the current device. Beware that an excessive allocation of pinned memory can slow down the host execution, as the program is left with less physical memory available to map the rest of the virtual addresses used.

- Device memory is allocated using `hipMalloc` which is later freed using `hipFree`

- With `hipMemcpy` data bytes can be transferred from host to device (using `hipMemcpyHostToDevice`) or from device to host (using `hipMemcpyDeviceToHost`), among others.

- `myKernelName<<<...>>>` queues the kernel execution on the device. All the kernels are launched on the `hipStreamDefault`, meaning that these executions are performed in order. `hipGetLastError` returns the last error produced by any runtime API call, allowing to check if any kernel launch resulted in error.

- `hipEventCreate` creates the events used to measure kernel execution time, `hipEventRecord` starts recording an event and  `hipEventSynchronize` waits for all the previous work in the stream when the specified event was recorded. With these three functions it can be measured the start and stop times of the kernel, and with `hipEventElapsedTime` the kernel execution time (in milliseconds) can be obtained.

## Demonstrated API Calls

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `__global__`
- `hipEventCreate`
- `hipEventDestroy`
- `hipEventElapsedTime`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipHostFree`
- `hipHostMalloc`
- `hipHostMallocMapped`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyDeviceToHost`
- `hipMemcpyHostToDevice`
- `hipStreamDefault`
