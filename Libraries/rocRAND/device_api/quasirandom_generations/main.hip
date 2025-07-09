// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocrand_utils.hpp"

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#include <iostream>
#include <vector>

///
/// \brief Initialize Sobol generator states.
///
/// Each thread takes care of a state.
///
/// \tparam Bits Length in number of bits of each generated number. Values should be either
///         32 (Sobol32) or 64 (Sobol64).
/// \tparam StateT Sobol generator state type.
///
/// \param states Sobol's generator states.
/// \param vectors Sobol's direction vectors.
/// \param offset Values to skip from the generated sequence.
///
template<unsigned int Bits, typename StateT>
__global__ void sobol_init_kernel(StateT* states, unsigned int* vectors, unsigned int offset = 0)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;

    StateT state;
    rocrand_init(&vectors[dimension * 32], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

///
/// \brief Initialize Scrambled Sobol generator states.
///
/// Each thread takes care of a state.
///
/// \tparam Bits Length in number of bits of each generated number. Values should be either
///         32 (ScrambledSobol32) or 64 (ScrambledSobol64).
/// \tparam StateT Scrambled Sobol generator state type.
///
/// \param states Scrambled Sobol's generator states.
/// \param vectors Scrambled Sobol's direction vectors.
/// \param scramble_constants Scrambled Sobol's constants.
/// \param offset Values to skip from the generated sequence.
///
template<unsigned int Bits, typename StateT>
__global__ void scrambled_sobol_init_kernel(StateT*       states,
                                            unsigned int* vectors,
                                            unsigned int* scramble_constants,
                                            unsigned int  offset = 0)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;

    StateT state;
    rocrand_init(&vectors[dimension * Bits],
                 scramble_constants[dimension],
                 offset + state_id,
                 &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

///
/// \brief Cooperative generation of \p size uniformly distributed random <tt>unsigned int</tt> with (Scrambled) Sobol.
///
/// The kernel is to be launched with a number of blocks per dimension, for a certain amount of
/// dimensions. \p size values need to be generated per dimension \p d in [0, gridDim.y - 1],
/// so all the threads from blocks (i , d) for \p i in [0, gridDim.x] will cooperatively generate
/// said values.
///
/// \tparam StateT Generator state type.
///
/// \param states Generators' states.
/// \param output Global memory output buffer.
/// \param size Amount of values to be generated.
///
template<typename StateT>
__global__ void generate_kernel(StateT* states, unsigned int* output, const size_t size)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = gridDim.x * blockDim.x;

    StateT       state  = states[gridDim.x * blockDim.x * dimension + state_id];
    const size_t offset = dimension * size;
    unsigned int index  = state_id;

    while(index < size)
    {
        output[offset + index] = rocrand(&state);
        skipahead(stride - 1, &state);
        index += stride;
    }
    state = states[gridDim.x * blockDim.x * dimension + state_id];
    skipahead(static_cast<unsigned int>(size), &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

int main()
{
    typedef rocrand_state_sobol32           sobol_state_t;
    typedef rocrand_state_scrambled_sobol32 scrambled_sobol_state_t;

    const unsigned int dimensions         = 2;
    const unsigned int size_per_dimension = 1000;
    const size_t output_size = dimensions * size_per_dimension; /*number of values to generate*/

    // 1. Allocate device output buffers.
    unsigned int* d_sobol_output;
    unsigned int* d_scrambled_sobol_output;

    HIP_CHECK(hipMalloc(&d_sobol_output, output_size * sizeof(*d_sobol_output)));
    HIP_CHECK(
        hipMalloc(&d_scrambled_sobol_output, output_size * sizeof(*d_scrambled_sobol_output)));

    HIP_CHECK(hipDeviceSynchronize());

    // 2. Allocate states.
    // Each thread handles one state.
    const unsigned int blocks_per_dimension = 2;
    const unsigned int threads              = 64;
    const unsigned int num_states           = dimensions * blocks_per_dimension * threads;

    // Sobol
    sobol_state_t* sobol_states;
    HIP_CHECK(hipMalloc(&sobol_states, num_states * sizeof(sobol_state_t)));

    // Scrambled Sobol
    scrambled_sobol_state_t* scrambled_sobol_states;
    HIP_CHECK(hipMalloc(&scrambled_sobol_states, num_states * sizeof(scrambled_sobol_state_t)));

    // 3. Get direction vectors.
    const unsigned int direction_vectors_size = 32 /*bits in generated numbers*/ * dimensions;

    // Sobol
    const unsigned int* h_sobol_vectors;
    unsigned int*       d_sobol_vectors;
    ROCRAND_CHECK(
        rocrand_get_direction_vectors32(&h_sobol_vectors, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    HIP_CHECK(hipMalloc(&d_sobol_vectors, direction_vectors_size * sizeof(*d_sobol_vectors)));
    HIP_CHECK(hipMemcpy(d_sobol_vectors,
                        h_sobol_vectors,
                        direction_vectors_size * sizeof(*d_sobol_vectors),
                        hipMemcpyHostToDevice));

    // Scrambled Sobol
    const unsigned int* h_scrambled_sobol_vectors;
    unsigned int*       d_scrambled_sobol_vectors;
    ROCRAND_CHECK(rocrand_get_direction_vectors32(&h_scrambled_sobol_vectors,
                                                  ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    HIP_CHECK(hipMalloc(&d_scrambled_sobol_vectors,
                        direction_vectors_size * sizeof(*d_scrambled_sobol_vectors)));
    HIP_CHECK(hipMemcpy(d_scrambled_sobol_vectors,
                        h_scrambled_sobol_vectors,
                        direction_vectors_size * sizeof(*d_scrambled_sobol_vectors),
                        hipMemcpyHostToDevice));

    // 4. For Scrambled Sobol, get scramble constants.
    const unsigned int* h_scramble_constants;
    unsigned int*       d_scramble_constants;
    ROCRAND_CHECK(rocrand_get_scramble_constants32(&h_scramble_constants));
    HIP_CHECK(hipMalloc(&d_scramble_constants, dimensions * sizeof(*d_scramble_constants)));
    HIP_CHECK(hipMemcpy(d_scramble_constants,
                        h_scramble_constants,
                        dimensions * sizeof(*d_scramble_constants),
                        hipMemcpyHostToDevice));

    HIP_CHECK(hipDeviceSynchronize());

    // 5. Generate values.
    // For each generator, we first initialize the state and then we generate the values.
    const dim3 grid_size(blocks_per_dimension, dimensions);
    const dim3 block_size(threads);

    // Sobol
    std::cout << "Initializing Sobol generator...";

    sobol_init_kernel<32><<<grid_size, block_size>>>(sobol_states, d_sobol_vectors);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "done. \nLaunching generate_kernel for Sobol...";

    generate_kernel<<<grid_size, block_size>>>(sobol_states, d_sobol_output, size_per_dimension);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Scrambled Sobol
    std::cout << "done. \nInitializing Scrambled Sobol generator...";

    scrambled_sobol_init_kernel<32><<<grid_size, block_size>>>(scrambled_sobol_states,
                                                               d_scrambled_sobol_vectors,
                                                               d_scramble_constants);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "done. \nLaunching generate_kernel for Scrambled Sobol...";

    generate_kernel<<<grid_size, block_size>>>(scrambled_sobol_states,
                                               d_scrambled_sobol_output,
                                               size_per_dimension);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "done. \nCopying results to host...";

    // 6. Copy outputs from device to host and free device memory.
    std::vector<unsigned int> h_sobol_output(output_size);
    HIP_CHECK(hipMemcpy(h_sobol_output.data(),
                        d_sobol_output,
                        output_size * sizeof(*d_sobol_output),
                        hipMemcpyDeviceToHost));

    std::vector<unsigned int> h_scrambled_sobol_output(output_size);
    HIP_CHECK(hipMemcpy(h_scrambled_sobol_output.data(),
                        d_scrambled_sobol_output,
                        output_size * sizeof(*d_scrambled_sobol_output),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_sobol_output));
    HIP_CHECK(hipFree(d_sobol_vectors));
    HIP_CHECK(hipFree(d_scrambled_sobol_output));
    HIP_CHECK(hipFree(d_scrambled_sobol_vectors));
    HIP_CHECK(hipFree(d_scramble_constants));

    std::cout << "done. \nValidating results...";

    // 7. Check uniformity of results.
    double mean = 0.f;

    // Sobol
    if((mean = is_not_uniform_dist(h_sobol_output)))
    {
        std::cout
            << "Validation failed! The normalized mean of the Sobol-generated values should be "
               "near 0.5 but it's "
            << mean << std::endl;
        return error_exit_code;
    }

    // Scrambled Sobol
    if((mean = is_not_uniform_dist(h_scrambled_sobol_output)))
    {
        std::cout << "Validation failed! The normalized mean of the Scrambled Sobol-generated "
                     "values should be near 0.5 but it's "
                  << mean << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation succeded!" << std::endl;
    return 0;
}