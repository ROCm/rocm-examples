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

// Reduction
#include <Reduction/v4.hpp>
#include <bench_utils.hpp>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <limits>

int main(int argc, char** argv)
{
    // Initialize Google Benchmark
    benchmark::Initialize(&argc, argv);

    // Initialize benchmarks
    reduction::select_device_or_exit(&argc, argv);
    reduction::add_common_benchmark_info();
    auto input_sizes = reduction::create_input_sizes();
    auto block_sizes = reduction::create_block_sizes();
    auto input       = reduction::create_input(input_sizes);

    // User-defined reduction
    auto          kernel_op = [] __device__(unsigned lhs, unsigned rhs) { return max(lhs, rhs); };
    reduction::v4 reduce{kernel_op, std::numeric_limits<unsigned>::min(), input_sizes, block_sizes};

    auto benchmarks = reduction::create_benchmarks(input_sizes, block_sizes, input, reduce);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
