// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipblaslt_utils.hpp"

int main()
{
    using namespace tensor_manipulation;

    constexpr size_t m{18};
    constexpr size_t k{34};
    auto             weight = tensor::create<int>({m, k});

    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < k; ++j)
        {
            weight.set_value<int>({i, j}, i * k + j);
        }
    }

    std::cout << "Original weight:\n";
    print_tensor_data_multi_dims<int>(std::cout, weight);

    constexpr size_t mi_m       = 16;
    constexpr size_t mi_k       = 16;
    constexpr size_t mi_kv      = 4;
    constexpr size_t pack_k     = 2;
    constexpr auto   multiple_m = mi_m;
    constexpr auto   multiple_k = mi_k * pack_k;
    const auto       padded_m   = (m / multiple_m + !!(m % multiple_m)) * multiple_m;
    const auto       padded_k   = (k / multiple_k + !!(k % multiple_k)) * multiple_k;
    shape_t          padded_shape{padded_m, padded_k};
    auto             padded_weight = pad_tensor<int>(weight, padded_shape, 0);

    std::cout << "Padded weight:\n";
    print_tensor_data_multi_dims<int>(std::cout, padded_weight);

    padded_weight.reshape(
        {padded_m / mi_m, mi_m, padded_k / (mi_k * pack_k), mi_k / mi_kv, mi_kv * pack_k});
    tensor permuted = permute_tensor<int>(padded_weight, {0, 2, 3, 1, 4});

    std::cout << "Swizzle weight:\n";
    print_tensor_data_multi_dims<int>(std::cout, permuted);

    return 0;
}
