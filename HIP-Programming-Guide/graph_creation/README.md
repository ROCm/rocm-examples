# HIP-Programming-Guide Graph Creation Example

## Description

HIP graphs are an alternative way of executing tasks on a GPU that can provide performance benefits over launching
kernels using the standard method via streams. A HIP graph is made up of nodes and edges. The nodes of a HIP graph
represent the operations performed, while the edges mark dependencies between those operations.

Graphs can be created directly using the HIP graph API, giving fine-grained control over the graph. In this case, the
graph nodes are created explicitly, together with their parameters and dependencies, which specify the edges of the
graph, thereby forming the graph structure.

This example demonstrates how to explicitly create HIP graphs.. It should be compared to the
[graph capture example](../graph_capture).

### Application flow

1. A data vector is created on the host.
2. An empty graph template is created.
3. Two graph nodes are created. Each represents the allocation of a single device buffer.
4. The device allocation nodes are added to the graph template.
5. A graph node is created. It represents a host-side function which initializes the host data vector.
6. The initialization node is added to the graph template.
7. A graph node is created and added to the template. It represents the copy of the host vector to one of the device
   buffers. Therefore, this node has two dependencies: one on the allocation node for the targeted device buffer and one
   on the host initialization node.
8. A graph node is created. It represents the launch of the first kernel. This kernel makes use of the memory copied
   from the host. Its node therefore has a dependency on the node created in the previous step.
9. The first kernel node is added to the graph template.
10. A graph node is created. It represents the launch of the second kernel. This kernel initializes the second device
   buffer. Its node therefore has a dependency on the second allocation node.
11. The second kernel node is added to the graph template.
12. A graph node is created and added to the template. It represents the launch of the third kernel. This kernel makes
    use of the two previous kernels' results. Its node therefore has a dependency on both previous kernel nodes.
13. A graph node is created and added to the template. It represents the copy of the third kernel's result to the host
    vector. It therefore has a dependency on the third kernel node.
14. A graph node is created and added to the template. It frees the memory of the first device buffer which holds the
    third kernel's result. It therefore depends on the device-to-host copy node.
15. A graph node is created and added to the template. It frees the memory of the second device buffer which holds input
    data for the third kernel. It therefore depends on the third kernel node.
16. The graph template is now complete. A runnable graph is instantiated.
17. The graph template is no longer required. It is destroyed.
18. The runnable graph is launched on a HIP stream.
19. The stream is synchronized with the host.
20. The results are verified.
21. The runnable graph and stream are destroyed.

## Key APIs and Concepts

* `hipGraphCreate` creates an empty graph template.
* `hipGraphAddMemAllocNode` adds a device memory allocation node to a template.
* `hipGraphAddHostNode` adds a host-side function node to a template.
* `hipGraphAddMemcpyNode1D` adds a 1D `memcpy` operation node to a template.
* `hipGraphAddKernelNode` adds a kernel node to a template.
* `hipGraphAddMemFreeNode` adds a node freeing device memory to a template.
* `hipGraphInstantiate` instantiates a runnable graph from a given template.
* `hipGraphDestroy` destroys a graph template.
* `hipGraphLaunch` launches a runnable graph.
* `hipGraphExecDestroy` destroys a runnable graph.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipGetErrorString`
* `hipGraphAddHostNode`
* `hipGraphAddKernelNode`
* `hipGraphAddMemAllocNode`
* `hipGraphAddMemFreeNode`
* `hipGraphAddMemcpyNode1D`
* `hipGraphCreate`
* `hipGraphDestroy`
* `hipGraphExecDestroy`
* `hipGraphInstantiate`
* `hipGraphLaunch`
* `hipStreamCreate`
* `hipStreamDestroy`
