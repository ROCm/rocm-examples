// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>
// Workaround for ROCm on Windows not including `__half` definitions, in a host compiler.
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP__) && (defined(WIN32) || defined(_WIN32))
    #include <hip/amd_detail/hip_fp16_gcc.h>
#endif
#include <rocrand/rocrand.hpp>

#include "argument_parsing.hpp"
#include "example_utils.hpp"

// An anonymous namespace sets static linkage to its contents.
// This means that the contained function definitions will only be visible
// in the current compilation unit (i.e. cpp source file).
namespace
{

/// \brief Selects the device (GPU) with the provided ID. If it cannot be selected
/// (e.g. a non-existent device ID is passed), an exception is thrown.
/// Otherwise, the name of the device is queried and printed to the standard output.
void set_device(const int device_id)
{
    HIP_CHECK(hipSetDevice(device_id));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, device_id));
    std::cout << "Device is set to \"" << properties.name << "\"" << std::endl;
}

/// \brief Generates a random vector of type \p T on the device (GPU) with the size of \p size
/// using random distribution \p Distribution.
/// \p Distribution must be a rocRAND distribution.
/// The data is generated into device memory, and then is copied to an \p std::vector.
template<typename T, typename Distribution>
std::vector<T> generate_random_vector_on_device(const size_t size)
{
    // Instantiating a rocRAND C++ engine object takes care of initialization.
    rocrand_cpp::default_random_engine engine;

    // The same is true about a rocRAND distribution.
    Distribution distribution;

    // Allocate the requested amount of device memory.
    T* device_vector{};
    HIP_CHECK(hipMalloc(&device_vector, size * sizeof(T)))

    // `operator()` of the distribution generates the requested count of random numbers
    // into the provided memory location, using the provided random engine.
    distribution(engine, device_vector, size);

    // Allocate host memory.
    std::vector<T> host_vector(size);

    // Copy the device memory to the host. This call synchronizes the device execution with the host's.
    HIP_CHECK(
        hipMemcpy(host_vector.data(), device_vector, size * sizeof(T), hipMemcpyDeviceToHost));

    // Free up the device memory allocated earlier.
    HIP_CHECK(hipFree(device_vector));

    return host_vector;
}

/// \brief Generates a random vector of type \p T on the host (CPU) with the size of \p size
/// using random distribution \p Distribution.
/// \p Distribution must satisfy the standard RandomNumberDistribution requirements.
template<typename T, typename Distribution>
std::vector<T> generate_random_vector_on_host(const size_t size)
{
    // Instantiate the standard default random engine.
    std::default_random_engine engine;
    // Instantiate the standard random distribution.
    Distribution distribution;
    // Allocate host memory.
    std::vector<T> host_vector(size);

    // `std::generate` calls the provided lambda for every element in the vector to set
    // the value of the element.
    std::generate(host_vector.begin(), host_vector.end(), [&]() { return distribution(engine); });

    return host_vector;
}

/// \brief Generates a random vector of type \p T with the size of \p size both on the device (GPU)
/// using the distribution \p DeviceDistribution and on the host (CPU) using the distribution
/// \p HostDistribution. If argument \p print is set, the generated values are printed to the
/// standard output. The time it takes to fill the vectors with random values is measured and printed
/// to the standard output for both the device and the host case.
template<typename T, typename DeviceDistribution, typename HostDistribution>
void compare_device_and_host_random_number_generation(const size_t size, const bool print)
{
    // This local function measures the time it takes to execute the passed function `operation`.
    // Also prints the resulting random vector if `print` is set to true.
    const auto measure_time_and_print = [=](const auto operation, const std::string_view title)
    {
        // Record the time before and after invoking `operation`.
        const auto start         = std::chrono::high_resolution_clock::now();
        const auto result_vector = operation();
        const auto end           = std::chrono::high_resolution_clock::now();

        // Convert and print the duration in milliseconds.
        const std::chrono::duration<double, std::milli> duration_ms = end - start;
        std::cout << title << " took " << duration_ms.count() << " ms" << std::endl;
        if(print)
        {
            // Print the list of space-delimited values
            for(const auto val : result_vector)
            {
                std::cout << val << ' ';
            }
            std::cout << std::endl;
        }
    };

    // Run and measure the random number generation on the device.
    measure_time_and_print(
        [size]() { return generate_random_vector_on_device<T, DeviceDistribution>(size); },
        "Random number generation on the device");

    // Run and measure the random number generation on the host.
    measure_time_and_print([size]()
                           { return generate_random_vector_on_host<T, HostDistribution>(size); },
                           "Random number generation on the host");
}

/// \brief Executes the random number generation on both the device (GPU) and host (CPU)
/// based on which \p Distribution was selected on the command line.
void dispatch_distribution_type(const Distribution dist, const size_t size, const bool print)
{
    // Based on the passed `Distribution`, select the appropriate template arguments
    // to invoke `compare_device_and_host_random_number_generation` with.
    switch(dist)
    {
        case Distribution::uniform_int:
            compare_device_and_host_random_number_generation<
                unsigned int,
                rocrand_cpp::uniform_int_distribution<unsigned int>,
                std::uniform_int_distribution<unsigned int>>(size, print);
            break;
        case Distribution::uniform_real:
            compare_device_and_host_random_number_generation<
                float,
                rocrand_cpp::uniform_real_distribution<float>,
                std::uniform_real_distribution<float>>(size, print);
            break;
        case Distribution::normal:
            compare_device_and_host_random_number_generation<
                double,
                rocrand_cpp::normal_distribution<double>,
                std::normal_distribution<double>>(size, print);
            break;
        case Distribution::poisson:
            compare_device_and_host_random_number_generation<
                unsigned int,
                rocrand_cpp::poisson_distribution<unsigned int>,
                std::poisson_distribution<unsigned int>>(size, print);
        default: break;
    }
}

} // namespace

int main(const int argc, const char** argv)
{
    CliArguments args;
    try
    {
        // Parsing command line arguments. If something unexpected happens (e.g. missing arguments or
        // wrong format), an exception is thrown.
        args = parse_args(argc, argv);
        // The parsed arguments are logged to the output to provide feedback to the user.
        // For implementation, see `std::ostream& operator<<(std::ostream& os, const CliArguments& cli_args)`
        std::cout << args << std::endl;
    }
    catch(const std::exception& ex)
    {
        // The exception is caught, and an error message and the command line help is printed.
        // The program returns with a non-zero exit code.
        std::cerr << "Could not parse arguments. Error: "s.append(ex.what()) << "\n"
                  << cli_usage_message << std::endl;
        return error_exit_code;
    }

    // Set up the used device (GPU) according to the command line supplied argument.
    set_device(args.device_id_);

    // Run the selected measurement on the device (GPU) and host (CPU).
    dispatch_distribution_type(args.distribution_, args.size_, args.print_);
}
