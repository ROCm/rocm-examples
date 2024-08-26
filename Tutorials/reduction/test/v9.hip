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
#include <Reduction/v0.hpp>
#include <Reduction/v9.hpp>
#include <test_utils.hpp>

// Google Test
#include <gtest/gtest.h>

// STL
#include <limits>

int main(int argc, char** argv)
{
    // Initialize Google Test
    testing::InitGoogleTest(&argc, argv);

    // Initialize tests
    reduction::select_device_or_exit(&argc, argv);
    auto block_sizes = reduction::create_block_sizes();
    auto input_sizes = reduction::create_input_sizes();
    auto item_counts = reduction::create_items_per_threads();
    auto input       = reduction::create_input(input_sizes);

    // User-defined reduction
    auto          host_op = [] __host__(unsigned lhs, unsigned rhs) { return std::max(lhs, rhs); };
    reduction::v0 ref{host_op, std::numeric_limits<unsigned>::min(), input_sizes, block_sizes};
    auto          kernel_op = [] __device__(unsigned lhs, unsigned rhs) { return max(lhs, rhs); };
    reduction::v9 reduce{kernel_op, std::numeric_limits<unsigned>::min(), input_sizes, block_sizes};

    reduction::register_tests(input_sizes, block_sizes, item_counts, input, reduce, ref);

    return RUN_ALL_TESTS();
}
