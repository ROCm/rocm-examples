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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Original implementation by Pawan Harish based on the paper:
// "Accelerating Large Graph Algorithms on the GPU using CUDA" (HiPC'07)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess)                 \
    {                                        \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
        std::exit(EXIT_FAILURE);             \
    }                                        \
}

#define MAX_THREADS_PER_BLOCK 512

// Structure to hold node information
struct Node {
  int starting;     // Starting index in edge list
  int no_of_edges;  // Number of edges from this node
};

// [Sphinx HIP BFS kernel 1 start]
__global__ void BFS_Kernel(Node *g_graph_nodes, int *g_graph_edges,
                           char *g_graph_mask, char *g_updating_graph_mask,
                           char *g_graph_visited, int *g_cost, int no_of_nodes) {
  int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
  
  if (tid < no_of_nodes && g_graph_mask[tid]) {
    g_graph_mask[tid] = 0;
    
    // Traverse all edges of this node
    for (int i = g_graph_nodes[tid].starting;
         i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting);
         i++) {
      int id = g_graph_edges[i];
      if (!g_graph_visited[id]) {
        g_cost[id] = g_cost[tid] + 1;
        g_updating_graph_mask[id] = 1;
      }
    }
  }
}
// [Sphinx HIP BFS kernel 1 end]

// [Sphinx HIP BFS kernel 2 start]
__global__ void BFS_Update(char *g_graph_mask, char *g_updating_graph_mask,
                           char *g_graph_visited, bool *g_over, int no_of_nodes) {
  int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
  
  if (tid < no_of_nodes && g_updating_graph_mask[tid]) {
    g_graph_mask[tid] = 1;
    g_graph_visited[tid] = 1;
    *g_over = true;
    g_updating_graph_mask[tid] = 0;
  }
}
// [Sphinx HIP BFS kernel 2 end]

void BFSGraph(int argc, char **argv);

int main(int argc, char **argv) {
  BFSGraph(argc, argv);
  return EXIT_SUCCESS;
}

void Usage(int argc, char **argv) {
  std::cerr << "Usage: " << argv[0] << " <input_file>\n";
}

void BFSGraph(int argc, char **argv) {
  if (argc != 2) {
    Usage(argc, argv);
    std::exit(EXIT_FAILURE);
  }

  char *input_f = argv[1];
  std::cout << "Reading graph file: " << input_f << std::endl;

  // Read in Graph from a file
  FILE *fp = fopen(input_f, "r");
  if (!fp) {
    std::cerr << "Error: Failed to open graph file: " << input_f << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int no_of_nodes = 0;
  fscanf(fp, "%d", &no_of_nodes);
  std::cout << "Number of nodes: " << no_of_nodes << std::endl;

  int num_of_blocks = 1;
  int num_of_threads_per_block = no_of_nodes;

  // Distribute threads across multiple blocks if necessary
  if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
    num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
    num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
  }

  // Allocate host memory
  std::vector<Node> h_graph_nodes(no_of_nodes);
  std::vector<char> h_graph_mask(no_of_nodes, 0);
  std::vector<char> h_updating_graph_mask(no_of_nodes, 0);
  std::vector<char> h_graph_visited(no_of_nodes, 0);

  // Read node information
  int start, edgeno;
  for (int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
  }

  // Read the source node
  int source = 0;
  fscanf(fp, "%d", &source);
  source = 0;  // Force source to 0

  // Set the source node as active
  h_graph_mask[source] = 1;
  h_graph_visited[source] = 1;

  // Read edge list
  int edge_list_size = 0;
  fscanf(fp, "%d", &edge_list_size);
  std::cout << "Number of edges: " << edge_list_size << std::endl;

  std::vector<int> h_graph_edges(edge_list_size);
  int id, cost;
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp, "%d %d", &id, &cost);
    h_graph_edges[i] = id;
  }

  fclose(fp);
  std::cout << "Graph loaded successfully" << std::endl;

  // Allocate memory for the result (cost array)
  std::vector<int> h_cost(no_of_nodes, -1);
  h_cost[source] = 0;

  std::cout << "\nAllocating GPU memory..." << std::endl;

  // Allocate device memory
  Node *d_graph_nodes;
  int *d_graph_edges;
  char *d_graph_mask;
  char *d_updating_graph_mask;
  char *d_graph_visited;
  int *d_cost;
  bool *d_over;

  HIP_CHECK(hipMalloc(&d_graph_nodes, sizeof(Node) * no_of_nodes));
  HIP_CHECK(hipMalloc(&d_graph_edges, sizeof(int) * edge_list_size));
  HIP_CHECK(hipMalloc(&d_graph_mask, sizeof(char) * no_of_nodes));
  HIP_CHECK(hipMalloc(&d_updating_graph_mask, sizeof(char) * no_of_nodes));
  HIP_CHECK(hipMalloc(&d_graph_visited, sizeof(char) * no_of_nodes));
  HIP_CHECK(hipMalloc(&d_cost, sizeof(int) * no_of_nodes));
  HIP_CHECK(hipMalloc(&d_over, sizeof(bool)));

  std::cout << "Copying data to GPU..." << std::endl;

  // Copy data to device
  HIP_CHECK(hipMemcpy(d_graph_nodes, h_graph_nodes.data(),
                      sizeof(Node) * no_of_nodes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_graph_edges, h_graph_edges.data(),
                      sizeof(int) * edge_list_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_graph_mask, h_graph_mask.data(),
                      sizeof(char) * no_of_nodes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_updating_graph_mask, h_updating_graph_mask.data(),
                      sizeof(char) * no_of_nodes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_graph_visited, h_graph_visited.data(),
                      sizeof(char) * no_of_nodes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_cost, h_cost.data(), sizeof(int) * no_of_nodes,
                      hipMemcpyHostToDevice));

  std::cout << "Starting BFS traversal..." << std::endl;
  std::cout << "Grid: " << num_of_blocks << " blocks, " 
            << num_of_threads_per_block << " threads per block" << std::endl;

  // Setup execution parameters
  dim3 grid(num_of_blocks, 1, 1);
  dim3 threads(num_of_threads_per_block, 1, 1);

  // Run BFS until convergence
  int k = 0;
  bool stop;

  do {
    stop = false;
    HIP_CHECK(hipMemcpy(d_over, &stop, sizeof(bool), hipMemcpyHostToDevice));

    // Launch kernel to process current frontier
    BFS_Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask,
                                   d_updating_graph_mask, d_graph_visited,
                                   d_cost, no_of_nodes);
    HIP_CHECK(hipGetLastError());

    // Launch kernel to update masks
    BFS_Update<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask,
                                   d_graph_visited, d_over, no_of_nodes);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(&stop, d_over, sizeof(bool), hipMemcpyDeviceToHost));
    k++;
  } while (stop);

  HIP_CHECK(hipDeviceSynchronize());

  std::cout << "BFS completed after " << k << " iterations" << std::endl;

  // Copy result back to host
  HIP_CHECK(hipMemcpy(h_cost.data(), d_cost, sizeof(int) * no_of_nodes,
                      hipMemcpyDeviceToHost));

  // Store the result
  std::cout << "Writing results to result.txt..." << std::endl;
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < no_of_nodes; i++) {
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  }
  fclose(fpo);
  std::cout << "Results saved to result.txt" << std::endl;

  // Calculate statistics
  int max_cost = 0;
  int reachable_nodes = 0;
  for (int i = 0; i < no_of_nodes; i++) {
    if (h_cost[i] >= 0) {
      reachable_nodes++;
      if (h_cost[i] > max_cost) {
        max_cost = h_cost[i];
      }
    }
  }

  std::cout << "\nBFS Statistics:" << std::endl;
  std::cout << "Source node: " << source << std::endl;
  std::cout << "Reachable nodes: " << reachable_nodes << " / " << no_of_nodes
            << std::endl;
  std::cout << "Maximum distance: " << max_cost << std::endl;

  std::cout << "\nExecution completed successfully." << std::endl;

  // Cleanup device memory
  HIP_CHECK(hipFree(d_graph_nodes));
  HIP_CHECK(hipFree(d_graph_edges));
  HIP_CHECK(hipFree(d_graph_mask));
  HIP_CHECK(hipFree(d_updating_graph_mask));
  HIP_CHECK(hipFree(d_graph_visited));
  HIP_CHECK(hipFree(d_cost));
  HIP_CHECK(hipFree(d_over));
}
