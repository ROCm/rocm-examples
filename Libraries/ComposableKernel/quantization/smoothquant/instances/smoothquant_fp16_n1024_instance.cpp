// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include "smoothquant_instance_common.hpp"

// clang-format off
//                                                  rm rn tm    tn  vn   pd    2p
#if 0
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  2,  4,  64, 8,  true ,false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  4,  4,  64, 4,  true ,false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  8,  4,  64, 2,  true ,false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1, 16,  4,  64, 1,  true ,false>>(const S&, A);

template float smoothquant_<trait_<ck_tile::fp16_t, 1,  1,  1, 256, 4,  true ,false>>(const S&, A);
#endif

template float smoothquant_<trait_<ck_tile::fp16_t,  1, 1, 2,  128, 8,  true, false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t,  1, 2, 2,  128, 4,  true, false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t,  1, 4, 2,  128, 2,  true, false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t,  1, 4, 1,  256, 1,  true, false>>(const S&, A);
// clang-format on
