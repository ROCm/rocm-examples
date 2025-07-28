# HIP-Programming-Guide Graph Capture Example

## Description

HIP graphs are an alternative way of executing tasks on a GPU that can provide performance benefits over launching
kernels using the standard method via streams. A HIP graph is made up of nodes and edges. The nodes of a HIP graph
represent the operations performed, while the edges mark dependencies between those operations.

The easy way to integrate HIP graphs into already existing code is to use `hipStreamBeginCapture()` and
`hipStreamEndCapture()` to obtain a `hipGraph_t` graph template that includes the captured operations.

This example demonstrates how to capture HIP streams with the HIP graph API. It should be compared to the
[graph creation example](../graph_creation).

### Application flow

1. A data vector is created on the host.
2. A stream is created and its operations are captured.
3. Two device buffers are created; the allocation operations are executed asynchronously by the stream.
4. A host function which initializes the host vector is placed in the stream.
5. The host vector's contents are copied to one of the device buffers; the copy operation is executed asynchronously by
   the stream.
6. Three kernels are placed in the stream and will be launched once the preceding stream operations have finished.
7. A copy operation which transfers the results to the host vector is placed in the stream.
8. The device memory is freed asynchronously by the stream.
9. Capturing of stream operations is stopped.
10. An executable HIP graph is created from the captured graph; the latter is destroyed afterwards.
11. The HIP graph is launched, repeating the stream operations from above.
12. The results are verified.
13. The graph and the stream are destroyed.

## Key APIs and Concepts

* `hipStreamBeginCapture` begins capturing the operations of a HIP stream.
* `hipStreamEndCapture` ends capturing the operations of a HIP stream.
* `hipGraphInstantiate` creates an executable HIP graph from a template (= captured stream operations).
* `hipGraphDestroy` destroys a graph template.
* `hipGraphLaunch` executes a graph of operations.
* `hipGraphExecDestroy` destroys an executable graph.

## Demonstrated API Calls

### HIP Runtime

#### Device symbols

* `blockDim`
* `blockIdx`
* `threadIdx`

#### Host symbols

* `hipFreeAsync`
* `hipGetErrorString`
* `hipGraphDestroy`
* `hipGraphExecDestroy`
* `hipGraphInstantiate`
* `hipGraphLaunch`
* `hipMallocAsync`
* `hipLaunchKernelGGL`
* `hipMemcpyAsync`
* `hipStreamBeginCapture`
* `hipStreamCreate`
* `hipStreamDestroy`
* `hipStreamEndCapture`
