# HIP-Doc KMeans Clustering Example

## Description

This example demonstrates KMeans clustering using CPU-GPU cooperative computing
with HIP. KMeans is a widely-used unsupervised machine learning algorithm for
partitioning data into k distinct clusters based on feature similarity.

This implementation showcases **hybrid computing**, where the GPU handles the
highly parallel membership assignment phase, while the CPU manages the serial
centroid update phase. This division of labor optimizes performance by
leveraging each processor's strengths.

For more information on HIP programming, please refer to the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).

### Application flow

1. Random dataset is generated with configurable dimensions and cluster count.
2. Initial centroids are randomly selected from the dataset.
3. Data is copied to GPU memory (persistent allocation for efficiency).
4. **Iterative refinement loop**:
   - **GPU Phase (Parallel)**: Calculate membership for each data point:
     - Each thread processes one data point
     - Computes Euclidean distance to all centroids
     - Assigns point to nearest centroid
   - Check convergence: If no membership changes, algorithm terminates
   - **CPU Phase (Serial)**: Update centroid positions:
     - Average the positions of all points in each cluster
     - Compute new centroid locations
5. Results are copied back from GPU to CPU.
6. Final statistics and cluster quality metrics are computed.
7. Results are saved to file.
8. GPU memory is freed.

### KMeans Algorithm

KMeans clustering works by:

1. **Initialization**: Select k initial centroids (cluster centers)
2. **Assignment**: Assign each point to the nearest centroid
3. **Update**: Move centroids to the mean position of assigned points
4. **Repeat**: Continue until centroids stabilize (convergence)

The algorithm minimizes within-cluster variance, creating compact, spherical
clusters.

## Key Concepts

### CPU-GPU Cooperative Computing

This example demonstrates **hybrid computing** where:

**GPU Handles (Parallel):**

- Distance calculations (highly data-parallel)
- Membership assignment (independent per point)
- Thousands of distance computations simultaneously

**CPU Handles (Serial):**

- Centroid updates (averaging/reduction)
- Convergence checking
- Result validation and statistics

**Why This Division?**

- GPU excels at independent, parallel operations
- CPU better for sequential operations and small reductions
- Data transfers between CPU-GPU are minimized

### Euclidean Distance Calculation

Distance between point and centroid:

$$
\text{distance} = \sqrt{\sum_{i} (\text{point}[i] - \text{centroid}[i])^2}
$$

In the kernel, we compute squared distance (avoid expensive sqrt):

```cpp
for (int d = 0; d < dimension; ++d) {
    float diff = point[d] - centroid[d];
    distance += diff * diff;
}
```

### Convergence Detection

The algorithm converges when:

- No points change cluster membership
- OR maximum iterations reached
- OR centroids move less than a threshold

### Memory Transfer Strategy

```bash
Initial: CPU data → GPU (once)
Each iteration:
  - CPU centroids → GPU (small)
  - GPU memberships → CPU (moderate)
Final: (already on CPU, no transfer needed)
```

This minimizes data movement by keeping the large dataset on the GPU throughout.

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
- `FLT_MAX`: Maximum float value (from CUDA/HIP)

### Machine Learning Concepts

- **Unsupervised learning**: No labeled training data
- **Clustering**: Grouping similar data points
- **Centroid**: Mean position of all points in a cluster
- **WCSS**: Within-Cluster Sum of Squares (quality metric, lower is better)

## Performance Considerations

### Parallel Efficiency

- **Distance calculations**: O(n × k × d) operations, highly parallel
- **Centroid updates**: O(n × d) operations, sequential but simple
- **Convergence**: Typically 10-50 iterations

### Optimization Opportunities

- **Shared memory**: Cache centroids in shared memory
- **Reduced precision**: Use FP16 for distance calculations
- **Vectorization**: Process multiple dimensions per thread
- **Batch processing**: Update memberships in chunks

### Scalability

- Scales well with data size (n)
- Less efficient with many clusters (k) due to sequential comparisons
- Dimension count (d) affects both CPU and GPU equally

## Building and Running

### Build with Make

```bash
make
```

### Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

### Run with Default Parameters

```bash
./hip_kmeans
# Default: 10000 points, 2 dimensions, 5 clusters, 100 max iterations
```

### Run with Custom Parameters

```bash
./hip_kmeans <num_points> <dimensions> <k_clusters> <max_iterations>

# Examples:
./hip_kmeans 50000 3 10 200    # 50k points, 3D, 10 clusters
./hip_kmeans 100000 2 8 150    # 100k points, 2D, 8 clusters
./hip_kmeans 5000 10 5 100     # 5k points, 10D, 5 clusters
```

## Example Output

```bash
KMeans Clustering Configuration:
================================
Data points: 10000
Dimensions: 2
Clusters (k): 5
Max iterations: 100

Generating random dataset...
Initializing centroids...
Starting KMeans iterations...

Iteration   1: 9995 points changed clusters
Iteration   2: 3421 points changed clusters
Iteration   3: 1203 points changed clusters
Iteration   4: 456 points changed clusters
Iteration   5: 189 points changed clusters
Iteration   6: 67 points changed clusters
Iteration   7: 23 points changed clusters
Iteration   8: 5 points changed clusters
Iteration   9: 0 points changed clusters - Converged!

KMeans converged after 9 iterations.

Final Results:
==============
Within-Cluster Sum of Squares (WCSS): 78234.56

Cluster Distribution:
  Cluster 0: 2134 points (21.3%)
  Cluster 1: 1876 points (18.8%)
  Cluster 2: 2056 points (20.6%)
  Cluster 3: 1998 points (20.0%)
  Cluster 4: 1936 points (19.4%)

Final Centroids:
  Cluster 0: [4.82, 5.13]
  Cluster 1: [14.91, 15.06]
  Cluster 2: [25.03, 24.87]
  Cluster 3: [34.78, 35.21]
  Cluster 4: [44.93, 45.08]

Saving results to kmeans_results.txt...

Execution completed successfully.
```

## Algorithm Analysis

### Time Complexity

- **Per iteration**: O(n × k × d)
  - n = number of points
  - k = number of clusters
  - d = number of dimensions
- **Total**: O(i × n × k × d) where i = iterations to convergence

### Space Complexity

- **Host**: O(n × d + k × d) for data and centroids
- **Device**: O(n × d + n + k × d) for data, memberships, centroids

### Convergence Properties

- Usually converges in 10-100 iterations
- Convergence depends on:
  - Initial centroid placement
  - Data distribution
  - Number of clusters (k)
  - Data dimensionality

### Quality Metrics

- **WCSS** (Within-Cluster Sum of Squares): Lower is better
- **Silhouette Score**: Measures cluster separation
- **Davies-Bouldin Index**: Lower indicates better clustering

## Limitations and Considerations

### Algorithm Limitations

1. **Assumes spherical clusters**: Works best with round, evenly-sized clusters
2. **Sensitive to initialization**: Different initial centroids → different results
3. **Requires k to be specified**: Must know number of clusters beforehand
4. **Outliers affect results**: Extreme values can distort centroids
5. **Local optima**: May not find globally optimal clustering

### Solutions

- **K-means++**: Better initialization strategy
- **Multiple runs**: Run several times, pick best result
- **Elbow method**: Determine optimal k
- **Preprocessing**: Remove outliers, normalize features

## Demonstrated API calls

### HIP runtime

#### Device symbols

- `blockIdx`
- `blockDim`
- `threadIdx`
- `FLT_MAX`

#### Host symbols

- `hipDeviceSynchronize`
- `hipFree`
- `hipGetLastError`
- `hipMalloc`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`

## Hybrid Computing Pattern

This example demonstrates the **CPU-GPU cooperation pattern**:

```cpp
// 1. Allocate GPU memory once (persistent)
hipMalloc(&gpuData, size);
hipMemcpy(gpuData, hostData, size, hipMemcpyHostToDevice);

// 2. Iterative loop
while (!converged) {
    // GPU: Parallel operation
    gpuKernel<<<grid, block>>>(gpuData, ...);
    hipMemcpy(results, gpuResults, size, hipMemcpyDeviceToHost);
    
    // CPU: Serial operation
    updateOnCPU(results);
    
    // Check convergence
    if (noChanges) break;
}

// 3. Cleanup
hipFree(gpuData);
```

## Applications

KMeans clustering is used in:

- **Image segmentation**: Group similar pixels
- **Customer segmentation**: Identify customer groups
- **Anomaly detection**: Find outliers
- **Data compression**: Vector quantization
- **Document clustering**: Group similar documents
- **Market research**: Identify market segments

## Future Enhancements

Possible improvements:

1. **K-means++**: Better initialization
2. **Mini-batch KMeans**: Process data in batches
3. **GPU centroid update**: Parallelize averaging
4. **Shared memory**: Cache centroids on chip
5. **Multiple runs**: Run several times, pick best
6. **Adaptive k**: Automatically determine cluster count
