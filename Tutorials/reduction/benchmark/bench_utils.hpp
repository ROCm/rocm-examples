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

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <chrono>
#include <cstdlib> // std::exit
#include <iostream> // std::cerr, std::endl
#include <numeric> // iota
#include <span>
#include <string> // std::stirng, std::to_string
#include <string_view>
#include <vector>

namespace reduction
{
void select_device_or_exit(int* argc = nullptr, char** argv = nullptr)
{
    int dev_id    = *argc > 1 ? std::atoi(argv[1]) : 0;
    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));

    if(dev_id < dev_count)
        HIP_CHECK(hipSetDevice(dev_id));
    else
    {
        std::cerr << "Unable to select device id: " << dev_id << "\n";
        std::cerr << "Number of available devices: " << dev_count << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void add_common_benchmark_info()
{
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    auto str = [](const std::string& name, const std::string& val)
    { benchmark::AddCustomContext(name, val); };

    auto num = [](const std::string& name, const auto& value)
    { benchmark::AddCustomContext(name, std::to_string(value)); };

    auto dim2 = [num](const std::string& name, const auto* values)
    {
        num(name + "_x", values[0]);
        num(name + "_y", values[1]);
    };

    auto dim3 = [num, dim2](const std::string& name, const auto* values)
    {
        dim2(name, values);
        num(name + "_z", values[2]);
    };

    str("hdp_name", devProp.name);
    num("hdp_total_global_mem", devProp.totalGlobalMem);
    num("hdp_shared_mem_per_block", devProp.sharedMemPerBlock);
    num("hdp_regs_per_block", devProp.regsPerBlock);
    num("hdp_warp_size", devProp.warpSize);
    num("hdp_max_threads_per_block", devProp.maxThreadsPerBlock);
    dim3("hdp_max_threads_dim", devProp.maxThreadsDim);
    dim3("hdp_max_grid_size", devProp.maxGridSize);
    num("hdp_clock_rate", devProp.clockRate);
    num("hdp_memory_clock_rate", devProp.memoryClockRate);
    num("hdp_memory_bus_width", devProp.memoryBusWidth);
    num("hdp_major", devProp.major);
    num("hdp_minor", devProp.minor);
    num("hdp_multi_processor_count", devProp.multiProcessorCount);
    num("hdp_l2_cache_size", devProp.l2CacheSize);
    num("hdp_max_threads_per_multiprocessor", devProp.maxThreadsPerMultiProcessor);
    num("hdp_compute_mode", devProp.computeMode);
    num("hdp_clock_instruction_rate", devProp.clockInstructionRate);
    num("hdp_concurrent_kernels", devProp.concurrentKernels);
    num("hdp_max_shared_memory_per_multi_processor", devProp.maxSharedMemoryPerMultiProcessor);
    str("hdp_gcn_arch_name", devProp.gcnArchName);
    num("hdp_integrated", devProp.integrated);
}

std::vector<std::size_t> create_input_sizes()
{
    return std::vector<std::size_t>{10,
                                    //50,
                                    100,
                                    //500,
                                    1'000,
                                    //5'000,
                                    10'000,
                                    //50'000,
                                    100'000,
                                    //500'000,
                                    1'000'000,
                                    //5'000'000,
                                    10'000'000,
                                    //50'000'000,
                                    100'000'000,
                                    //500'000'000,
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

template<typename T, typename Reduction>
std::vector<benchmark::internal::Benchmark*>
    create_benchmarks(const std::span<std::size_t> input_sizes,
                      const std::span<std::size_t> block_sizes,
                      const std::span<std::size_t> item_counts,
                      const std::vector<T>&        input,
                      Reduction&                   reduce)
{
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    for(const auto input_size : input_sizes)
        for(const auto block_size : block_sizes)
            std::transform(
                item_counts.begin(),
                item_counts.end(),
                std::back_inserter(benchmarks),
                [&](const std::size_t item_count)
                {
                    auto bench = benchmark::RegisterBenchmark(
                        (std::to_string(input_size) + "_" + std::to_string(block_size) + "_"
                         + std::to_string(item_count))
                            .c_str(),
                        [&reduce, &input, input_size, block_size, item_count](
                            benchmark::State& state)
                        {
                            // Run (length of loop comes from `b->Iterations(run_count);`)
                            for(auto _ : state)
                            {
                                auto [result, elapsed_milliseconds]
                                    = reduce(std::span<const T>{input.data(), input_size},
                                             block_size,
                                             item_count);

                                benchmark::DoNotOptimize(result);

                                state.SetIterationTime(
                                    std::chrono::duration_cast<std::chrono::duration<double>>(
                                        elapsed_milliseconds)
                                        .count());
                            }
                            state.SetBytesProcessed(state.iterations() * input_size * sizeof(T));
                            state.SetItemsProcessed(state.iterations() * input_size);
                        });

                    bench->UseManualTime();
                    bench->Unit(benchmark::kMillisecond);

                    return bench;
                });
    return benchmarks;
}

template<typename T, typename Reduction>
std::vector<benchmark::internal::Benchmark*>
    create_benchmarks(const std::span<std::size_t> input_sizes,
                      const std::span<std::size_t> block_sizes,
                      const std::vector<T>&        input,
                      Reduction&                   reduce)
{
    std::vector<std::size_t> temp{1};
    return create_benchmarks(input_sizes, block_sizes, temp, input, reduce);
}
} // namespace reduction
