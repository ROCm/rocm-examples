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

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <cfloat>

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

// [Sphinx HIP KMeans membership kernel start]
__global__ void updateMembershipGPU(float *data, float *centroids,
                                    int *membership, int dataSize, int dimension,
                                    int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dataSize) {
    float minDistance = FLT_MAX;
    int nearestCluster = 0;

    // Find the nearest centroid for this data point
    for (int cluster = 0; cluster < k; ++cluster) {
      float distance = 0.0f;

      // Calculate Euclidean distance
      for (int d = 0; d < dimension; ++d) {
        float diff = data[idx * dimension + d] - centroids[cluster * dimension + d];
        distance += diff * diff;
      }

      if (distance < minDistance) {
        minDistance = distance;
        nearestCluster = cluster;
      }
    }

    membership[idx] = nearestCluster;
  }
}
// [Sphinx HIP KMeans membership kernel end]

// Initialize centroids randomly from the dataset
std::vector<float> initializeCentroids(const std::vector<float> &data, int length,
                                       int dimension, int k) {
  std::vector<float> centroids(k * dimension);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, length - 1);

  // Select k random data points as initial centroids
  for (int i = 0; i < k; ++i) {
    int randomIdx = dis(gen);
    for (int d = 0; d < dimension; ++d) {
      centroids[i * dimension + d] = data[randomIdx * dimension + d];
    }
  }

  return centroids;
}

// Update membership using GPU (hybrid CPU-GPU function)
int updateMembership(float *gpuData, std::vector<float> &centroids,
                     std::vector<int> &membership, int dataSize, int dimension,
                     int k) {
  float *gpuCentroids;
  int *gpuMembership;

  // Allocate GPU memory
  HIP_CHECK(hipMalloc(&gpuCentroids, k * dimension * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpuMembership, dataSize * sizeof(int)));

  // Copy centroids from CPU to GPU
  HIP_CHECK(hipMemcpy(gpuCentroids, centroids.data(),
                      k * dimension * sizeof(float), hipMemcpyHostToDevice));

  // Calculate the sizes for the kernel launch
  int localSize = 256;
  int globalSize = (dataSize + localSize - 1) / localSize;

  // Launch the kernel
  updateMembershipGPU<<<globalSize, localSize>>>(
      gpuData, gpuCentroids, gpuMembership, dataSize, dimension, k);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Create CPU data to hold the results
  std::vector<int> cpuNewMembership(dataSize);

  // Copy GPU data back to CPU
  HIP_CHECK(hipMemcpy(cpuNewMembership.data(), gpuMembership,
                      dataSize * sizeof(int), hipMemcpyDeviceToHost));

  // Count membership updates
  int membershipUpdate = 0;
  for (int i = 0; i < dataSize; ++i) {
    if (membership[i] != cpuNewMembership[i]) {
      membershipUpdate++;
      membership[i] = cpuNewMembership[i];
    }
  }

  // Free GPU memory
  HIP_CHECK(hipFree(gpuCentroids));
  HIP_CHECK(hipFree(gpuMembership));

  return membershipUpdate;
}

// Update centroids on CPU (serial portion)
void updateCentroid(const std::vector<float> &data,
                    std::vector<float> &centroids,
                    const std::vector<int> &membership, int dataSize,
                    int dimension, int k) {
  std::vector<int> clusterSizes(k, 0);
  std::vector<float> newCentroids(k * dimension, 0.0f);

  // Sum all points in each cluster
  for (int i = 0; i < dataSize; ++i) {
    int cluster = membership[i];
    clusterSizes[cluster]++;

    for (int d = 0; d < dimension; ++d) {
      newCentroids[cluster * dimension + d] += data[i * dimension + d];
    }
  }

  // Calculate the average (mean) for each cluster
  for (int cluster = 0; cluster < k; ++cluster) {
    if (clusterSizes[cluster] > 0) {
      for (int d = 0; d < dimension; ++d) {
        centroids[cluster * dimension + d] =
            newCentroids[cluster * dimension + d] / clusterSizes[cluster];
      }
    }
  }
}

// Generate random dataset
std::vector<float> generateRandomData(int length, int dimension, int k) {
  std::vector<float> data(length * dimension);
  std::random_device rd;
  std::mt19937 gen(rd());

  // Create k clusters with different centers
  for (int cluster = 0; cluster < k; ++cluster) {
    // Random center for this cluster
    std::vector<float> center(dimension);
    std::uniform_real_distribution<> centerDis(cluster * 10.0f,
                                               (cluster + 1) * 10.0f);
    for (int d = 0; d < dimension; ++d) {
      center[d] = centerDis(gen);
    }

    // Generate points around this center
    int pointsPerCluster = length / k;
    std::normal_distribution<> pointDis(0.0, 2.0);  // Spread around center

    for (int i = 0; i < pointsPerCluster; ++i) {
      int idx = cluster * pointsPerCluster + i;
      if (idx >= length)
        break;

      for (int d = 0; d < dimension; ++d) {
        data[idx * dimension + d] = center[d] + pointDis(gen);
      }
    }
  }

  // Fill any remaining points
  std::uniform_real_distribution<> fillDis(0.0f, k * 10.0f);
  for (int i = (length / k) * k; i < length; ++i) {
    for (int d = 0; d < dimension; ++d) {
      data[i * dimension + d] = fillDis(gen);
    }
  }

  return data;
}

// Calculate total within-cluster sum of squares (quality metric)
float calculateWCSS(const std::vector<float> &data,
                    const std::vector<float> &centroids,
                    const std::vector<int> &membership, int dataSize,
                    int dimension, int k) {
  float wcss = 0.0f;

  for (int i = 0; i < dataSize; ++i) {
    int cluster = membership[i];
    float distance = 0.0f;

    for (int d = 0; d < dimension; ++d) {
      float diff = data[i * dimension + d] - centroids[cluster * dimension + d];
      distance += diff * diff;
    }

    wcss += distance;
  }

  return wcss;
}

// Save results to file
void saveResults(const std::string &filename, const std::vector<float> &centroids,
                 const std::vector<int> &membership, int k, int dimension) {
  std::ofstream outfile(filename);

  outfile << "Final Centroids:\n";
  for (int i = 0; i < k; ++i) {
    outfile << "Cluster " << i << ": [";
    for (int d = 0; d < dimension; ++d) {
      outfile << centroids[i * dimension + d];
      if (d < dimension - 1)
        outfile << ", ";
    }
    outfile << "]\n";
  }

  outfile << "\nCluster Sizes:\n";
  std::vector<int> clusterSizes(k, 0);
  for (int m : membership) {
    clusterSizes[m]++;
  }

  for (int i = 0; i < k; ++i) {
    outfile << "Cluster " << i << ": " << clusterSizes[i] << " points\n";
  }

  outfile.close();
}

int main(int argc, char **argv) {
  // Default parameters
  int length = 10000;      // Number of data points
  int dimension = 2;       // Number of features per point
  int k = 5;               // Number of clusters
  int maxIterations = 100; // Maximum iterations

  // Parse command line arguments
  if (argc > 1)
    length = std::atoi(argv[1]);
  if (argc > 2)
    dimension = std::atoi(argv[2]);
  if (argc > 3)
    k = std::atoi(argv[3]);
  if (argc > 4)
    maxIterations = std::atoi(argv[4]);

  std::cout << "KMeans Clustering Configuration:\n";
  std::cout << "================================\n";
  std::cout << "Data points: " << length << "\n";
  std::cout << "Dimensions: " << dimension << "\n";
  std::cout << "Clusters (k): " << k << "\n";
  std::cout << "Max iterations: " << maxIterations << "\n\n";

  // Generate random dataset
  std::cout << "Generating random dataset...\n";
  std::vector<float> data = generateRandomData(length, dimension, k);

  // Copy data to GPU (persistent allocation)
  float *gpuData;
  HIP_CHECK(hipMalloc(&gpuData, length * dimension * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpuData, data.data(), length * dimension * sizeof(float),
                      hipMemcpyHostToDevice));

  // Initialize centroids and membership
  std::cout << "Initializing centroids...\n";
  std::vector<float> centroids = initializeCentroids(data, length, dimension, k);
  std::vector<int> membership(length, 0);

  std::cout << "Starting KMeans iterations...\n\n";

  // Main KMeans loop
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    // GPU: Determine the cluster that each entry belongs to
    int membershipChanges =
        updateMembership(gpuData, centroids, membership, length, dimension, k);

    std::cout << "Iteration " << std::setw(3) << (iteration + 1) << ": "
              << membershipChanges << " points changed clusters";

    // Convergence checking
    if (membershipChanges == 0) {
      std::cout << " - Converged!\n";
      std::cout << "\nKMeans converged after " << (iteration + 1)
                << " iterations.\n";
      break;
    }
    std::cout << "\n";

    // CPU: Calculate new centroids
    updateCentroid(data, centroids, membership, length, dimension, k);
  }

  // Calculate quality metric
  float wcss = calculateWCSS(data, centroids, membership, length, dimension, k);
  std::cout << "\nFinal Results:\n";
  std::cout << "==============\n";
  std::cout << "Within-Cluster Sum of Squares (WCSS): " << std::fixed
            << std::setprecision(2) << wcss << "\n";

  // Count cluster sizes
  std::vector<int> clusterSizes(k, 0);
  for (int m : membership) {
    clusterSizes[m]++;
  }

  std::cout << "\nCluster Distribution:\n";
  for (int i = 0; i < k; ++i) {
    std::cout << "  Cluster " << i << ": " << clusterSizes[i] << " points ("
              << std::fixed << std::setprecision(1)
              << (100.0 * clusterSizes[i] / length) << "%)\n";
  }

  // Display final centroids
  std::cout << "\nFinal Centroids:\n";
  for (int i = 0; i < k; ++i) {
    std::cout << "  Cluster " << i << ": [";
    for (int d = 0; d < dimension; ++d) {
      std::cout << std::fixed << std::setprecision(2)
                << centroids[i * dimension + d];
      if (d < dimension - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }

  // Save results
  std::cout << "\nSaving results to kmeans_results.txt...\n";
  saveResults("kmeans_results.txt", centroids, membership, k, dimension);

  // Cleanup
  HIP_CHECK(hipFree(gpuData));

  std::cout << "\nExecution completed successfully.\n";

  return EXIT_SUCCESS;
}
