# HIP-Doc Breadth-First Search (BFS) Example

## Description

This example demonstrates Breadth-First Search (BFS) on a GPU using HIP, based
on the algorithm presented in the HiPC'07 paper "Accelerating Large Graph
Algorithms on the GPU using CUDA" by Pawan Harish and P.J. Narayanan.

BFS is a fundamental graph traversal algorithm that explores nodes level by
level, starting from a source node. This parallel implementation efficiently
processes large graphs by assigning one thread per node, allowing thousands of
nodes to be processed simultaneously.

For more information on HIP programming, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).

### Application flow

1. A graph is loaded from a text file containing node and edge information.
2. Host memory is allocated for node structures, edge lists, and traversal state.
3. The source node (node 0) is marked as the starting point.
4. Device memory is allocated for all graph data structures.
5. Graph data is copied from host to device memory.
6. The BFS kernel is launched iteratively until no more nodes can be visited:
   - **Kernel 1 (BFS_Kernel)**: Processes all nodes in the current frontier:
     - Each thread handles one node
     - If a node is in the frontier, it explores all neighbors
     - Unvisited neighbors are marked for the next iteration
     - Distance/cost is updated for newly discovered nodes
   - **Kernel 2 (BFS_Update)**: Updates the frontier for the next iteration:
     - Nodes discovered in this iteration become the new frontier
     - Visited flags are updated to prevent revisiting
     - A stop flag indicates if any new nodes were discovered
7. The process repeats until no new nodes are found (convergence).
8. Results are copied back to host memory.
9. The final costs (distances from source) are written to a file.
10. All device memory is freed.

### Graph Format

The input graph file format is:

```text
<number_of_nodes>
<starting_edge_index> <number_of_edges>   // For each node
...
<source_node>
<total_number_of_edges>
<destination_node> <edge_weight>          // For each edge
...
```

## Key Concepts

### Breadth-First Search Algorithm

BFS explores a graph level by level:

1. Start with a source node at distance 0
2. Visit all neighbors of the source (distance 1)
3. Visit all unvisited neighbors of distance-1 nodes (distance 2)
4. Continue until all reachable nodes are visited

### Parallel BFS on GPU

The parallel implementation uses a **frontier-based approach**:

- **Frontier**: The set of nodes to be processed in the current iteration
- **Two-kernel design**:
  1. Process current frontier and discover new nodes
  2. Update frontier for next iteration
- **One thread per node**: Each thread is responsible for processing one node
- **Iterative execution**: Kernels repeat until the frontier is empty

### Graph Representation

- **Nodes**: Store the starting index in the edge list and number of edges
- **Edges**: Flat array of destination node IDs
- **Masks and Flags**:
  - `graph_mask`: Current frontier (nodes to process this iteration)
  - `updating_graph_mask`: Nodes discovered this iteration (next frontier)
  - `graph_visited`: Tracks all visited nodes (prevents revisiting)
  - `over`: Flag to indicate if any new nodes were discovered

### Memory Layout

The graph uses a Compressed Sparse Row (CSR) format:

```text
Nodes: [Node 0] [Node 1] [Node 2] ...
       ↓
Edges: [2,3,5] [1,4] [0,6,7] ...
```

Each node stores where its edges start in the edge array and how many edges it has.

## Key APIs and Concepts

### HIP Runtime APIs

- `hipMalloc`: Allocates device memory
- `hipMemcpy`: Transfers data between host and device
- `hipFree`: Frees device memory
- `hipGetLastError`: Retrieves the last error from a runtime call
- `hipDeviceSynchronize`: Blocks until all device operations complete

### Device Code Features

- `__global__`: Declares a kernel function callable from host
- `blockIdx`, `blockDim`, `threadIdx`: Built-in variables for grid/block indexing
- Iterative kernel execution until convergence

### Graph Algorithms

- **Frontier-based traversal**: Efficiently processes only active nodes
- **Level synchronization**: All nodes at distance d are processed before distance d+1
- **Distance computation**: Computes shortest path distances (in number of hops)

## Performance Considerations

### Parallel Efficiency

- **Work distribution**: One thread per node provides good load balancing
- **Memory coalescing**: Threads access memory in patterns that can be coalesced
- **Irregular parallelism**: Graph structure affects parallel efficiency

### Optimization Opportunities

- **Warp-level optimization**: Could use warp-level primitives to reduce divergence
- **Load balancing**: Large-degree nodes can cause load imbalance
- **Memory layout**: Could optimize edge list layout for better cache utilization
- **Early termination**: Could terminate when a specific target node is reached

## Building and Running

### Build with Make

```bash
make
```

This builds both:

- `hip_bfs`: The BFS application
- `graphgen`: Graph generator utility

### Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

### Generate Test Graphs

```bash
./graphgen 1024 1k      # Creates graph1k.txt with 1024 nodes
./graphgen 4096 4k      # Creates graph4k.txt with 4096 nodes
./graphgen 65536 64k    # Creates graph64k.txt with 65536 nodes
```

### Run BFS

```bash
./hip_bfs graph4096.txt
```

### Output Files

- `result.txt`: Contains the distance from source to each node

## Example Output

```text
Reading graph file: graph4096.txt
Number of nodes: 4096
Number of edges: 49152
Graph loaded successfully

Allocating GPU memory...
Copying data to GPU...
Starting BFS traversal...
Grid: 8 blocks, 512 threads per block
BFS completed after 12 iterations
Writing results to result.txt...
Results saved to result.txt

BFS Statistics:
Source node: 0
Reachable nodes: 4096 / 4096
Maximum distance: 11

Execution completed successfully.
```

## Algorithm Analysis

### Time Complexity

- **Sequential**: O(V + E) where V = vertices, E = edges
- **Parallel**: O(D × (V/P + E/P)) where D = graph diameter, P = processors
- **Best case**: O(D) when V, E >> P (high parallelism)

### Space Complexity

- O(V + E) for graph storage
- O(V) for traversal state (masks, costs, visited flags)

### Convergence

The algorithm terminates when no new nodes are discovered, which occurs after D
iterations where D is the graph diameter (longest shortest path).

## Demonstrated API calls

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`

#### Host symbols

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`

## References

- Harish, P., & Narayanan, P. J. (2007). "Accelerating Large Graph Algorithms on
  the GPU using CUDA." In International Conference on High Performance Computing
  (HiPC).

## Graph Generator

The included `graphgen` utility creates random graphs for testing:

- Generates undirected graphs
- Random number of edges per node (2-4)
- Edge weights between 1-10
- Output format compatible with the BFS application
