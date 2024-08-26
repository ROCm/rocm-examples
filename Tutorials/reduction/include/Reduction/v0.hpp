
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

// STL
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace reduction
{
template<typename T, typename BinaryOperator>
class v0
{
public:
    v0(const BinaryOperator& host_op_in,
       const T               zero_elem_in,
       const std::span<size_t>,
       const std::span<size_t>)
        : host_op{host_op_in}, zero_elem{zero_elem_in}
    {}

    std::tuple<T, std::chrono::duration<float, std::milli>>
        operator()(std::span<const T> input, const std::size_t, const std::size_t)
    {
        std::chrono::high_resolution_clock::time_point start, end;

        start       = std::chrono::high_resolution_clock::now();
        auto result = std::reduce(std::execution::par_unseq,
                                  input.begin(),
                                  input.end(),
                                  zero_elem,
                                  host_op);
        end         = std::chrono::high_resolution_clock::now();

        return {result,
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start)};
    }

private:
    BinaryOperator host_op;
    T              zero_elem;
};
} // namespace reduction
