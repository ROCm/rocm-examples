// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

/// \brief Transposes the matrix \p in and stores the result in \p out using warp shuffle operations.
__global__ void matrix_transpose_kernel(float* out, const float* in, const unsigned int width)
{
    // Compute the row and column indexes of the matrix element that each thread is going
    // to process. Since in this example there is only one block, the indexes are
    // precisely the thread's ID in each dimension.
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;

    // If not out of bounds, transpose element.
    if(x < width && y < width)
    {
        // Read element from global memory. Each thread in the warp is reading the element that
        // the thread with global id x * width + y will transpose.
        const float val = in[y * width + x];

        // Transpose element reading it from the correspondent thread with a shuffle operation:
        // * when targeting AMD devices, __shfl is used.
        // * with targeting NVIDIA devices, __shfl is deprecated from CUDA 9 and __shfl_sync must
        //   be used instead.
        // __shfl/___shfl_sync do not require all threads to be active, so they can be inside the
        // if block.
        // Note that, since the matrix in this example has less elements than the warp size value,
        // the ID within the warp of each thread matches its global ID.
        // Also note that __shfl_sync needs the mask of active threads to be explicitly passed as
        // argument. It can be retrieved with the warp intrinsic __activemask().
#if defined(__HIP_PLATFORM_AMD__)
        out[x * width + y] = __shfl(val, y * width + x);
#elif defined(__HIP_PLATFORM_NVIDIA__)
        out[x * width + y] = __shfl_sync(__activemask(), val, y * width + x);
#endif
    }
}

/// \brief CPU implementation of matrix transpose.
std::vector<float> expected_matrix_transpose(const std::vector<float>& input,
                                             const unsigned int        width)
{
    std::vector<float> output(width * width);
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
    return output;
}

int main()
{
    // Number of rows and columns, total number of elements and size in bytes of the matrix
    // to be transposed.
    constexpr unsigned int width      = 4;
    constexpr unsigned int size       = width * width;
    constexpr unsigned int size_bytes = size * sizeof(float);

    // Get device's warp size.
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0 /*device ID*/));

    // To guarantee the correct behaviour of the program, keep total number of matrix elements
    // below (or equal to) warp size.
    if(size > static_cast<unsigned int>(props.warpSize))
    {
        std::cerr << "Matrix has more elements than architecture's warp size value." << std::endl;
        exit(error_exit_code);
    }

    // Block (2D) and grid sizes. Note that in this example we have only 1 block (and 1 warp).
    const dim3 block_dim(width, width);
    const dim3 grid_dim(1);

    // Allocate host input matrix and initialize with increasing sequence 10, 20, 30, ....
    std::vector<float> matrix(size);
    std::iota(matrix.begin(), matrix.end(), 1.f);
    std::for_each(matrix.begin(), matrix.end(), [](float& f) { f = 10.f * f; });

    // Allocate matrix to store the results of the kernel execution.
    std::vector<float> transposed_matrix(size);

    // Allocate input and output matrices on device.
    float* d_matrix{};
    float* d_transposed_matrix{};
    HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
    HIP_CHECK(hipMalloc(&d_transposed_matrix, size_bytes));

    // Copy input matrix data from host to device.
    HIP_CHECK(hipMemcpy(d_matrix, matrix.data(), size_bytes, hipMemcpyHostToDevice));

    // Print trace message.
    std::cout << "Computing matrix transpose." << std::endl;

    // Lauching kernel from host
    matrix_transpose_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_transposed_matrix,
                                                                          d_matrix,
                                                                          width);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy results from device to host.
    HIP_CHECK(hipMemcpy(transposed_matrix.data(),
                        d_transposed_matrix,
                        size_bytes,
                        hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_matrix));
    HIP_CHECK(hipFree(d_transposed_matrix));

    // Calculate expected transposed matrix with the CPU version of the kernel.
    std::vector<float> expected_transposed_matrix = expected_matrix_transpose(matrix, width);

    // Validate results comparing with expected transposed matrix.
    unsigned int    errors = 0;
    constexpr float eps    = 1.0E-6f;
    std::cout << "Validating transposed matrix." << std::endl;
    for(unsigned int i = 0; i < size; i++)
    {
        errors += (std::fabs(transposed_matrix[i] - expected_transposed_matrix[i]) > eps);
    }

    if(errors)
    {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
}
