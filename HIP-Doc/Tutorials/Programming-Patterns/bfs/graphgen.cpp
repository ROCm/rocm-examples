// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Original implementation by Sam Kauffman - University of Virginia
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

#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

#define MIN_NODES 20
#define MAX_NODES ULONG_MAX
#define MIN_EDGES 2
#define MAX_INIT_EDGES 4  // Nodes will have, on average, 2*MAX_INIT_EDGES edges
#define MIN_WEIGHT 1
#define MAX_WEIGHT 10

typedef unsigned int uint;
typedef unsigned long ulong;

struct edge {
  ulong dest;
  uint weight;
};

typedef vector<edge> node;

int main(int argc, char **argv) {
  // Parse command line
  ulong numNodes;
  string s;
  
  if (argc < 2) {
    cerr << "Error: enter a number of nodes.\n";
    cerr << "Usage: " << argv[0] << " <num_nodes> [<filename_suffix>]\n";
    exit(1);
  }
  
  numNodes = strtoul(argv[1], NULL, 10);
  if (numNodes < MIN_NODES || numNodes > MAX_NODES || argv[1][0] == '-') {
    cerr << "Error: Invalid argument: " << argv[1] << "\n";
    exit(1);
  }
  
  s = argc > 2 ? argv[2] : argv[1];  // filename suffix
  string filename = "graph" + s + ".txt";

  cout << "Generating graph with " << numNodes << " nodes...\n";
  node *graph;
  graph = new node[numNodes];

  // Initialize random number generators
  srand(time(NULL));
  mt19937_64 gen(time(NULL));
  uniform_int_distribution<ulong> randNode(0, numNodes - 1);

  // Generate graph
  uint numEdges;
  ulong nodeID;
  uint weight;
  
  for (ulong i = 0; i < numNodes; i++) {
    numEdges = rand() % (MAX_INIT_EDGES - MIN_EDGES + 1) + MIN_EDGES;
    for (uint j = 0; j < numEdges; j++) {
      nodeID = randNode(gen);
      weight = rand() % (MAX_WEIGHT - MIN_WEIGHT + 1) + MIN_WEIGHT;
      
      graph[i].push_back(edge());
      graph[i].back().dest = nodeID;
      graph[i].back().weight = weight;
      
      graph[nodeID].push_back(edge());
      graph[nodeID].back().dest = i;
      graph[nodeID].back().weight = weight;
    }
  }

  // Output
  cout << "Writing to file \"" << filename << "\"...\n";
  ofstream outf(filename);
  outf << numNodes << "\n";
  
  ulong totalEdges = 0;
  for (uint i = 0; i < numNodes; i++) {
    numEdges = graph[i].size();
    outf << totalEdges << " " << numEdges << "\n";
    totalEdges += numEdges;
  }
  
  outf << "\n" << randNode(gen) << "\n\n";
  outf << totalEdges << "\n";
  
  for (ulong i = 0; i < numNodes; i++) {
    for (uint j = 0; j < graph[i].size(); j++) {
      outf << graph[i][j].dest << " " << graph[i][j].weight << "\n";
    }
  }
  
  outf.close();
  cout << "Graph generated successfully.\n";
  cout << "Nodes: " << numNodes << ", Edges: " << totalEdges << "\n";

  delete[] graph;
  return 0;
}
