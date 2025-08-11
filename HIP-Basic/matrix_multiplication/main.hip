// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>

/// \brief Multiplies matrices \p A and \p B and stores the result to \p C.
/// - The number of rows of the result matrix is equal to the number of rows of matrix A
///   which is \p blockDim.y*gridDim.y.
/// - The number of columns of the result matrix is equal to the number of columns of matrix B
///   which is \p blockDim.x*gridDim.x.
/// - The number of columns of matrix \p A is passed as argument.
/// - The matrix elements are stored in a row-major order.
///
/// - Each thread in the grid is responsible for one element of the result matrix.
/// - Each element is calculated cooperatively in a tiled manner. In each step, a BlockSize*BlockSize
///   tile is loaded to the shared memory so individual threads can address this shared cache instead
///   of loading the same values from the global device memory individually.
///   The end result is accumulated through each step on a per-thread basis.
/// - The matrix dimensions are assumed to be multiples of the block size for simplicity.
template<unsigned int BlockSize>
__global__ void matrix_multiplication_kernel(const float*       A,
                                             const float*       B,
                                             float*             C,
                                             const unsigned int a_cols)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    // b_cols must match the number of output matrix columns.
    const unsigned int b_cols = blockDim.x * gridDim.x;

    // The number of tiles is determined by A's columns (which is equal to B's rows).
    const unsigned int steps = a_cols / BlockSize;

    // thread_result is the accumulation variable.
    float thread_result = 0.0F;
    for(unsigned int step = 0; step < steps; step++)
    {
        // Shared memory is used to cache the tile from both input matrices.
        // The tile is a square of BlockSize*BlockSize.
        __shared__ float a_values[BlockSize][BlockSize];
        __shared__ float b_values[BlockSize][BlockSize];

        // Index of the top-left element of the tile in A.
        // "BlockSize * a_cols * by" is the number of elements to move "down".
        // "BlockSize * step" is the number of elements to move "right".
        const unsigned int a_idx = BlockSize * (a_cols * by + step);

        // Index of the top-left element of the tile in B.
        // "BlockSize * b_cols * step" is the number of elements to move "down".
        // "BlockSize * bx" is the number of elements to move "right".
        const unsigned int b_idx = BlockSize * (b_cols * step + bx);

        // Load each element in the tile to shared memory.
        a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
        b_values[ty][tx] = B[b_idx + b_cols * ty + tx];

        // Synchronization is needed to make sure that all elements are loaded before
        // starting the calculation.
        __syncthreads();

        // Each thread calculates the scalar product of the tile and increments the
        // thread-individual thread_result.
        for(unsigned int i = 0; i < BlockSize; i++)
        {
            thread_result += a_values[ty][i] * b_values[i][tx];
        }

        // Synchronize to ensure that the calculation is finished before the next tile's
        // elements start to load.
        __syncthreads();
    }

    // Calculate the index of the top-left element of the output block.
    const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

    // Every thread stores the final result to global memory.
    C[block_offset + b_cols * ty + tx] = thread_result;
}
template<unsigned int BlockSize>
void configure_parser(cli::Parser& parser)
{
    // Default parameters
    constexpr unsigned int a_rows = 2048;
    constexpr unsigned int a_cols = 1024;
    constexpr unsigned int b_cols = 1024;

    static_assert(
        ((a_rows % BlockSize == 0) && (a_cols % BlockSize == 0) && (b_cols % BlockSize == 0)),
        "Matrix dimensions must be positive multiples of block_size");

    parser.set_optional<unsigned int>("A_rows",
                                      "A_rows",
                                      a_rows,
                                      "Number of rows in Matrix A"); // Default 2048
    parser.set_optional<unsigned int>("A_cols",
                                      "A_cols",
                                      a_cols,
                                      "Number of columns in Matrix A"); // Default 1024
    parser.set_optional<unsigned int>("B_cols",
                                      "B_cols",
                                      b_cols,
                                      "Number of columns in Matrix B"); // Default 1024
}

int main(int argc, const char* argv[])
{
    constexpr unsigned int block_size = 16;

    // Parse user inputs
    cli::Parser parser(argc, argv);
    configure_parser<block_size>(parser);
    parser.run_and_exit_if_error();

    // Get matrix dimensions from the command line, if provided.
    const unsigned int a_rows = parser.get<unsigned int>("A_rows");
    const unsigned int a_cols = parser.get<unsigned int>("A_cols");
    const unsigned int b_cols = parser.get<unsigned int>("B_cols");

    if((a_rows % block_size != 0) || (a_cols % block_size != 0) || (b_cols % block_size != 0))
    {
        std::cout << "Matrix dimensions must be positive multiples of block_size ("
                         + std::to_string(block_size) + ")"
                  << std::endl;
        exit(error_exit_code);
    }

    // Outer matrix dimensions must match.
    const unsigned int b_rows = a_cols;
    const unsigned int c_cols = b_cols;
    const unsigned int c_rows = a_rows;

    std::vector<float> A(a_cols * a_rows);
    std::vector<float> B(b_cols * b_rows);
    std::vector<float> C(c_cols * c_rows);

    // Set matrix elements to a constant on the host.
    std::fill(A.begin(), A.end(), 1.F);

    constexpr float b_value = 0.02F;
    std::fill(B.begin(), B.end(), b_value);

    const size_t a_bytes = sizeof(float) * A.size();
    const size_t b_bytes = sizeof(float) * B.size();
    const size_t c_bytes = sizeof(float) * C.size();
    float*       d_A{};
    float*       d_B{};
    float*       d_C{};
    HIP_CHECK(hipMalloc(&d_A, a_bytes));
    HIP_CHECK(hipMalloc(&d_B, b_bytes));
    HIP_CHECK(hipMalloc(&d_C, c_bytes));

    HIP_CHECK(hipMemcpy(d_A, A.data(), a_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), b_bytes, hipMemcpyHostToDevice));

    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(c_cols / block_size, c_rows / block_size);

    // Launch matrix multiplication kernel.
    std::cout << "Matrix multiplication: [" << a_rows << 'x' << a_cols << "] * [" << b_rows << 'x'
              << b_cols << "], block size: " << block_size << 'x' << block_size << std::endl;
    matrix_multiplication_kernel<block_size>
        <<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_A, d_B, d_C, a_cols);
    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the resulting matrix to the host. This call synchronizes with the host.
    HIP_CHECK(hipMemcpy(C.data(), d_C, c_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // Check if the resulting elements match the expectation.
    constexpr float tolerance         = 0.001F;
    const bool      validation_passed = std::all_of(
        C.begin(),
        C.end(),
        [=](const float value) { return tolerance > std::abs(value - a_cols * b_value); });
    if(validation_passed)
    {
        std::cout << "Validation passed." << std::endl;
    }
    else
    {
        std::cout << "Validation failed." << std::endl;
        return error_exit_code;
    }
}
