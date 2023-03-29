// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ostream>
#include <random>
#include <string>
#include <vector>

// clang-format off
/// \brief Convolution filter using arbitrary values
const constexpr std::array<float, 5 * 5> convolution_filter_5x5 = {1.0f,  3.0f, 0.0f,  -2.0f, -0.0f, 
                                                                   1.0f,  4.0f, 0.0f,  -8.0f, -4.0f,
                                                                   2.0f,  7.0f, 0.0f, -12.0f, -0.0f,
                                                                   2.0f,  3.0f, 1.5f,  -8.0f, -4.0f,
                                                                   0.0f,  1.0f, 0.0f,  -2.0f, -0.0f};
// clang-format on

/// \brief allocate memory in constant address space for the mask on the device
__constant__ float d_mask[5 * 5];

/// \brief Implements a convolution for an input grid \p input and a \p d_mask that is defined in constant memory. The \p input needs
/// to be padded such that \p mask_size is taken into account, i.e. padded_width = floor(mask_width/2) * 2 + width
/// and padded_height = floor(mask_height/2) * 2 + height
template<size_t MaskWidth = 5>
__global__ void convolution(const float* input, float* output, const uint2 input_dimensions)
{
    const size_t x            = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y            = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t width        = input_dimensions.x;
    const size_t height       = input_dimensions.y;
    const size_t padded_width = width + (MaskWidth / 2) * 2;

    // Check if the currently computed element is inside the grid domain.
    if(x >= width || y >= height)
        return;

    // Temporary storage variables.
    float        sum              = 0.0f;
    const size_t convolution_base = y * padded_width + x;

    // Iterate over the mask in both x and y direction.
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t mask_index         = mask_index_y * MaskWidth + mask_index_x;
            const size_t convolution_offset = mask_index_y * padded_width + mask_index_x;
            sum += input[convolution_base + convolution_offset] * d_mask[mask_index];
        }
    }

    output[y * width + x] = sum;
}

template<typename T>
void print_grid(std::vector<T> vec, int width)
{
    size_t num_rows = vec.size() / width;
    auto   it       = vec.begin();
    for(size_t i = 0; i < num_rows; i++)
    {
        std::copy(it, it + width, std::ostream_iterator<T>(std::cout, " "));
        std::cout << std::endl;
        it += width;
    }
}

/// \brief Reference CPU implementation of convolution for results verification.
template<typename mask_type>
void convolution_reference(std::vector<float>&       verificationOutput,
                           const std::vector<float>& paddedInput,
                           const mask_type&          mask,
                           const unsigned int        height,
                           const unsigned int        width,
                           const unsigned int        mask_width)
{
    // padded_width = width + floor(mask_width / 2) * 2
    const unsigned int padded_width = width + (mask_width / 2) * 2;
    // Iterate over the provided grid.
    for(unsigned int y = 0; y < height; y++)
    {

        for(unsigned int x = 0; x < width; x++)
        {
            // temporary for summation.
            float sum = 0.0f;
            // Iterate over the mask for the given element.
            for(unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y)
            {
                for(unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x)
                {
                    unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                    unsigned int input_index
                        = (y + mask_index_y) * padded_width + (x + mask_index_x);
                    sum += paddedInput[input_index] * mask[mask_index];
                }
            }
            verificationOutput[(y * width + x)] = sum;
        }
    }
}

/// \brief Adds to a command line parser the necessary options for this example.
template<unsigned int BlockSize>
void configure_parser(cli::Parser& parser)
{
    // Default parameters.
    const constexpr unsigned int width      = 4096;
    const constexpr unsigned int height     = 4096;
    const constexpr unsigned int iterations = 10;
    const constexpr bool         print      = false;

    parser.set_optional<unsigned int>("x", "width", width, "Width of the input grid");
    parser.set_optional<unsigned int>("y", "height", height, "Height of the input grid");
    parser.set_optional<unsigned int>("i",
                                      "iterations",
                                      iterations,
                                      "Number of times the algorithm is executed.");
    parser.set_optional<bool>("p", "print", print, "Enables printing the convoluted grid");
}

int main(int argc, char* argv[])
{
    // Number of threads in each kernel block dimension.
    const constexpr unsigned int block_size = 32;
    const constexpr unsigned int mask_width = 5;

    // Parse user input.
    cli::Parser parser(argc, argv);
    configure_parser<block_size>(parser);
    parser.run_and_exit_if_error();

    // Get number of nodes and iterations from the command line, if provided.
    const unsigned int width      = parser.get<unsigned int>("x");
    const unsigned int height     = parser.get<unsigned int>("y");
    const unsigned int iterations = parser.get<unsigned int>("i");
    const bool         print      = parser.get<bool>("p");

    // Check values provided.
    if(width < 1)
    {
        std::cout << "Width  must be at least 1. (provided " << width << " )" << std::endl;
        return error_exit_code;
    }
    if(height < 1)
    {
        std::cout << "Height  must be at least 1. (provided " << height << " )" << std::endl;
        return error_exit_code;
    }
    if(iterations < 1)
    {
        std::cout << "Iterations  must be at least 1. (provided " << iterations << " )"
                  << std::endl;
        return error_exit_code;
    }

    // Total number of elements and bytes of the input grid.
    const unsigned int size       = width * height;
    const unsigned int size_bytes = size * sizeof(float);

    const constexpr unsigned int mask_element_num = mask_width * mask_width;
    const constexpr unsigned int mask_size_bytes  = mask_element_num * sizeof(float);
    const constexpr unsigned int filter_radius    = mask_width / 2;

    const unsigned int padded_width            = width + filter_radius * 2;
    const unsigned int padded_height           = height + filter_radius * 2;
    const unsigned int input_size_padded       = padded_width * padded_height;
    const unsigned int input_size_padded_bytes = input_size_padded * sizeof(float);

    auto mask = convolution_filter_5x5;

    // Allocate host input grid initialized with random floats between 0-256.
    std::vector<float>                    input_grid(size);
    std::mt19937                          mersenne_engine{0};
    std::uniform_real_distribution<float> distribution{0, 256};
    auto                                  rnd = std::bind(distribution, mersenne_engine);
    std::generate(input_grid.begin(), input_grid.end(), rnd);

    // Allocate output grid.
    std::vector<float> output_grid(size);

    // Allocate padded input with zero boundary condition.
    std::vector<float> input_grid_padded(input_size_padded, 0);

    auto input_grid_row_begin = input_grid.begin();
    auto padded_input_grid_row_begin
        = input_grid_padded.begin() + filter_radius * padded_width + filter_radius;
    for(unsigned int i = 0; i < height; i++)
    {
        std::copy(input_grid_row_begin, input_grid_row_begin + width, padded_input_grid_row_begin);
        padded_input_grid_row_begin += padded_width;
        input_grid_row_begin += width;
    }

    // Allocate host memory for the CPU implementation and copy input data.
    std::vector<float> expected_output_grid(output_grid);

    std::cout << "Executing a simple convolution for " << iterations << " iterations with a "
              << width << " x " << height << " sized grid." << std::endl;

    // Allocate device memory.
    float* d_input_grid_padded;
    float* d_output_grid;

    HIP_CHECK(hipMalloc(&d_input_grid_padded, input_size_padded_bytes));
    HIP_CHECK(hipMalloc(&d_output_grid, size_bytes));

    // Copy input data from host to device memory.
    HIP_CHECK(hipMemcpy(d_input_grid_padded,
                        input_grid_padded.data(),
                        input_size_padded_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyToSymbol(d_mask, mask.data(), mask_size_bytes));

    // Cumulative variable to compute the mean bandwidth per iteration of the algorithm.
    double kernel_bandwidths = 0;

    // Cumulative variable to compute the mean time per iteration of the algorithm.
    double kernel_time = 0;

    // Create events to measure the execution time of the kernels.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Number of threads in each kernel block and number of blocks in the grid.
    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim((width + block_size) / block_size, (height + block_size) / block_size);

    // Run iterations times the convolution GPU algorithm.
    for(unsigned int i = 0; i < iterations; ++i)
    {
        float kernel_ms{};

        // Record the start event.
        HIP_CHECK(hipEventRecord(start, hipStreamDefault));

        // Launch Convolution kernel on the default stream.
        convolution<mask_width><<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_input_grid_padded,
                                                                              d_output_grid,
                                                                              {width, height});

        // Check if the kernel launch was successful.
        HIP_CHECK(hipGetLastError());

        // Record the stop event and wait until the kernel execution finishes.
        HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
        HIP_CHECK(hipEventSynchronize(stop));

        // Get the execution time of the kernel and add it to the total count.
        HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
        kernel_time += kernel_ms;
        kernel_bandwidths += (size_bytes + input_size_padded_bytes) / kernel_ms;
    }

    // Destroy hipEvents.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Copy results back to host.
    HIP_CHECK(hipMemcpy(output_grid.data(), d_output_grid, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_input_grid_padded));
    HIP_CHECK(hipFree(d_output_grid));

    // Print the mean time per iteration (in miliseconds) of the algorithm, and the estimated mean bandwidth in (GB/s).
    double average_bandwidth = kernel_bandwidths / iterations;
    kernel_time /= iterations;
    std::cout << "The mean time needed for each iteration has been " << kernel_time
              << "ms and mean bandwidth was " << average_bandwidth / 1e6 << " GB/s" << std::endl;

    // Execute CPU algorithm.
    convolution_reference(expected_output_grid, input_grid_padded, mask, height, width, mask_width);

    // Print the calculated grids.
    if(print)
    {
        std::cout << "Input grid:" << std::endl;
        print_grid(input_grid, width);
        std::cout << "Result grid:" << std::endl;
        print_grid(output_grid, width);
        std::cout << "CPU reference grid:" << std::endl;
        print_grid(expected_output_grid, width);
    }

    // Verify results.
    double error = 0;
    std::cout << "Validating results with CPU implementation." << std::endl;
    for(unsigned int i = 0; i < size; ++i)
    {
        double diff = (output_grid[i] - expected_output_grid[i]);
        error += diff * diff;
    }
    error = std::sqrt(error / size);
    std::cout << "The root-mean-square error of the difference between the reference and the gpu "
                 "result is "
              << error << std::endl;
}
