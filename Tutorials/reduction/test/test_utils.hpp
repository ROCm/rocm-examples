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

#pragma once

// Reduction
#include <hip_utils.hpp>

// HIP API
#include <hip/hip_runtime.h> // hipGetErrorString

// Google Test
#include <gtest/gtest.h>

// STL
#include <chrono>
#include <cstdlib> // std::exit
#include <initializer_list>
#include <iostream> // std::cerr, std::endl
#include <numeric> // std::iota, std::accumulate
#include <span>
#include <string> // std::stirng, std::to_string
#include <vector>

namespace reduction
{
void select_device_or_exit(int* argc = nullptr, char** argv = nullptr)
{
    int dev_id    = *argc > 1 ? std::atoi(argv[1]) : 0;
    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning( \
        disable : 4996) // This function or variable may be unsafe. Consider using _dupenv_s instead.
#endif
    static const std::string rg0 = "CTEST_RESOURCE_GROUP_0";
    if(std::getenv(rg0.c_str()) != nullptr)
    {
        std::string amdgpu_target = std::getenv(rg0.c_str());
        std::transform(
            amdgpu_target.cbegin(),
            amdgpu_target.cend(),
            amdgpu_target.begin(),
            // Feeding std::toupper plainly results in implicitly truncating conversions between int and char triggering warnings.
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        std::string reqs = std::getenv((rg0 + "_" + amdgpu_target).c_str());
        dev_id           = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
    }
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
    if(dev_id < dev_count)
        HIP_CHECK(hipSetDevice(dev_id));
    else
    {
        std::cerr << "Unable to select device id: " << dev_id << "\n";
        std::cerr << "Number of available devices: " << dev_count << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::vector<std::size_t> create_input_sizes()
{
    return std::vector<std::size_t>{10,
                                    50,
                                    100,
                                    500,
                                    1'000,
                                    5'000,
                                    10'000,
                                    50'000,
                                    100'000,
                                    500'000,
                                    1'000'000,
                                    5'000'000,
                                    10'000'000,
                                    50'000'000,
                                    100'000'000,
                                    500'000'000,
                                    1'000'000'000};
}

enum block_size_strategy
{
    include_warp_size = 0,
    exclude_warp_size
};

std::vector<std::size_t> create_block_sizes(block_size_strategy strategy = include_warp_size)
{
    std::vector<std::size_t> block_sizes;
    hipDeviceProp_t          props;
    int                      dev_id;
    HIP_CHECK(hipGetDevice(&dev_id));
    HIP_CHECK(hipGetDeviceProperties(&props, dev_id));
    for(auto block_size = (strategy == include_warp_size ? props.warpSize : props.warpSize * 2);
        block_size <= props.maxThreadsPerBlock;
        block_size *= 2)
    {
        block_sizes.push_back(block_size);
    }

    return block_sizes;
}

std::vector<std::size_t> create_items_per_threads()
{
    return std::vector<std::size_t>{1, 2, 3, 4, 8, 16};
}

std::vector<unsigned> create_input(const std::span<std::size_t> input_sizes)
{
    std::vector<unsigned> input(*std::max_element(input_sizes.begin(), input_sizes.end()));
    std::iota(input.begin(), input.end(), 0);

    return input;
}

class empty_fixture : public ::testing::Test
{
public:
    static void  SetUpTestSuite() {}
    static void  TearDownTestSuite() {}
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

template<typename T, typename Reduction, typename Reference>
class test_adapter : public empty_fixture
{
public:
    explicit test_adapter(const std::vector<T>& input_in,
                          const std::size_t     input_size_in,
                          const std::size_t     block_size_in,
                          const std::size_t     items_per_thread_in,
                          Reduction&            reduce_in,
                          Reference&            reference_in)
        : input{input_in}
        , input_size{input_size_in}
        , block_size{block_size_in}
        , items_per_thread{items_per_thread_in}
        , reduce{reduce_in}
        , reference{reference_in}
    {}

    virtual void TestBody() override
    {
        auto [result, elapsed_milliseconds]
            = reduce(std::span<const T>{input.data(), input_size}, block_size, items_per_thread);

        auto [ref_result, ref_elapsed_milliseconds]
            = reference(std::span<const T>{input.data(), input_size}, block_size, items_per_thread);

        ASSERT_EQ(result, ref_result);
    }

private:
    const std::vector<T>& input;
    const std::size_t     input_size;
    const std::size_t     block_size;
    const std::size_t     items_per_thread;
    Reduction&            reduce;
    Reference&            reference;
};

template<typename T, typename Reduction, typename Reference>
void register_tests(const std::span<std::size_t> input_sizes,
                    const std::span<std::size_t> block_sizes,
                    const std::span<std::size_t> items_per_threads,
                    const std::vector<T>&        input,
                    Reduction&                   reduce,
                    Reference&                   reference)
{
    for(const auto input_size : input_sizes)
        for(const auto block_size : block_sizes)
            for(const auto items_per_thread : items_per_threads)
                testing::RegisterTest(
                    "Reduction",
                    (std::to_string(input_size) + "_" + std::to_string(block_size) + "_"
                     + std::to_string(items_per_thread))
                        .c_str(),
                    nullptr,
                    std::to_string(0).c_str(),
                    __FILE__,
                    __LINE__,
                    [&, input_size, block_size, items_per_thread]() -> empty_fixture* {
                        return new test_adapter{input,
                                                input_size,
                                                block_size,
                                                items_per_thread,
                                                reduce,
                                                reference};
                    });
}
} // namespace reduction
