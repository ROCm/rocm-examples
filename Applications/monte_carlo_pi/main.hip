// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hiprand_utils.hpp"

#include <hipcub/device/device_reduce.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>
#include <hipcub/iterator/discard_output_iterator.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>
#include <hiprand/hiprand.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

/// \brief Given a sample's index, return 1 if the sample, for which both dimensions lie in
///        (0, 1], is contained within the disk centered at the origin with radius 1. Else return 0.
struct conversion_op
{
    /// \brief The number of samples is given by \p s, <tt>2 * s</tt> random numbers are given in
    ///        \p d_d, which first stores \p s x-values followed by \p s y-values.
    conversion_op(int s, float* d_d) : sample_count(s), d_data(d_d) {}

    __device__ __host__ __forceinline__ int operator()(const int& a) const
    {
        float x = d_data[a];
        float y = d_data[sample_count + a];
        // sample points are in (0, 1]
        float distance = x * x + y * y;
        return static_cast<int>(distance <= 1.f);
    }

    int    sample_count;
    float* d_data;
};

/// \brief Given <tt>2 * sample_count</tt> random numbers in \p d_data that are in (0, 1],
///        approximate pi with the assumption that the random numbers are uniformly distributed.
///        \p d_data first stores \p sample_count x-values followed by \p sample_count y-values.
float calculate_pi(int sample_count, float* d_data)
{
    // 4. Set up the input and output iterator for hipCUB's Sum.

    // Represents the samples' index.
    auto input_counting = hipcub::CountingInputIterator<int>(0);

    // Converts the sample's index to a 0 or 1, indicating whether the sample lies within the disk.
    conversion_op convert_op(sample_count, d_data);
    auto input = hipcub::TransformInputIterator<bool, conversion_op, decltype(input_counting)>(
        input_counting,
        convert_op);

    int* d_output{};
    HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

    // 5. Call hipCUB's Sum to calculate the required memory size, allocate that amount.
    void*       tmp_storage{};
    std::size_t tmp_storage_size{};

    HIP_CHECK(
        hipcub::DeviceReduce::Sum(tmp_storage, tmp_storage_size, input, d_output, sample_count));

    HIP_CHECK(hipMalloc(&tmp_storage, tmp_storage_size));

    // 6. Call hipCUB's Sum to calculate the number of samples within the circle.
    HIP_CHECK(
        hipcub::DeviceReduce::Sum(tmp_storage, tmp_storage_size, input, d_output, sample_count));

    // 7. Copy back the result and approximate pi.
    int num_items{};
    HIP_CHECK(hipMemcpy(&num_items, d_output, sizeof(int), hipMemcpyDeviceToHost));

    float pi = 4.f * num_items / sample_count;

    HIP_CHECK(hipFree(tmp_storage));

    return pi;
}

/// \brief Prints the time elapsed and the calculated value of pi with an error value.
void print_results(int                sample_count,
                   float              pi_calc,
                   float              elapsed_ms,
                   const std::string& random_kind)
{
    constexpr float pi = 3.14159265358979323846f; // ground truth

    float err = std::abs((pi_calc - pi) / pi * 100.f);
    std::cout << "Calculating pi using " << sample_count << " " << std::setw(6) << random_kind
              << "random samples: " << std::fixed << pi_calc << " (error: " << err
              << "%), which took " << elapsed_ms << " ms." << std::defaultfloat << std::endl;
}

int main(int argc, char* argv[])
{
    // 1. Parse user inputs.
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("s", "sample_count", 1u << 20, "Number of samples.");
    parser.run_and_exit_if_error();

    const int sample_count = parser.get<int>("s");
    if(sample_count <= 0)
    {
        std::cerr << "Sample count should be greater than 0." << std::endl;
        return error_exit_code;
    }

    // The samples have two dimensions, so two random numbers are required per sample.
    const int rng_count = 2 * sample_count;

    // 2. Allocate data, initialize variables.
    float* d_data{};
    HIP_CHECK(hipMalloc(&d_data, rng_count * sizeof(float)));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    float pi_calc{};
    float elapsed_ms{};

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    {
        // 3. Initialize hipRAND's default pseudo-random generator and generate 2 * n samples.
        hiprandGenerator_t gen;
        HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
        HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 42));

        HIPRAND_CHECK(hiprandGenerateUniform(gen, d_data, rng_count));

        // 4. - 7.
        pi_calc = calculate_pi(sample_count, d_data);

        // 8. Clean up the generator and print the result.
        HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    }
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    print_results(sample_count, pi_calc, elapsed_ms, "pseudo");

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    {
        // 9. Initialize hipRAND's default quasirandom generator, set the dimensions to two,
        //    and generate 2 * n samples.
        hiprandGenerator_t gen;
        HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_QUASI_DEFAULT));
        HIPRAND_CHECK(hiprandSetQuasiRandomGeneratorDimensions(gen, 2));

        // The first dimension will be in the first half of the array, the second dimension in the
        // second half.
        HIPRAND_CHECK(hiprandGenerateUniform(gen, d_data, rng_count));

        // 4. - 7.
        pi_calc = calculate_pi(sample_count, d_data);

        // 8. Clean up the generator and print the result.
        HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    }
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    print_results(sample_count, pi_calc, elapsed_ms, "quasi");

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipFree(d_data));
}
