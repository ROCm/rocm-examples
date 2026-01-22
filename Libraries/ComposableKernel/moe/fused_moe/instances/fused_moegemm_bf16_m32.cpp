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

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// clang-format off
template float fused_moegemm_<
    fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 0, 0, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 0, 1, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 0, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 1, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);
// clang-format on
