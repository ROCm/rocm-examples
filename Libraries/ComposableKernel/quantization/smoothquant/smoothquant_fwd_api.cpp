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

#include "smoothquant.hpp"

#include <ck_tile/core.hpp>

#include <stdexcept>

template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kTwoPass_>
using trait_ = smoothquant_traits_<DataType_,
                                   Repeat_M_,
                                   Repeat_N_,
                                   ThreadPerBlock_M_,
                                   ThreadPerBlock_N_,
                                   Vector_N_,
                                   kPadN_,
                                   kTwoPass_>;

template <typename data_type>
float smoothquant_dispatch(smoothquant_traits /*t*/,
                           smoothquant_args a,
                           const ck_tile::stream_config& s)
{
    float r = -1;
    // clang-format off
    //                                         rm  rn  tm  tn  vn   pd    2p
    if(a.n <= 64) {
            r = smoothquant_<trait_<data_type, 1,  1,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.n <= 128) {
        if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type, 1,  1,  4,  64, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type, 1,  2,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.n <= 256) {
        if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 1,  4,  64, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 2,  4,  64, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 4,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.n <= 512) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 1,  4,  64, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 2,  4,  64, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 4,  4,  64, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 8,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.n <= 768) {
        if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 3,  4,  64, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 6,  4,  64, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1,12,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.n <= 1024) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 1, 2,  128, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 2,  128, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 4, 2,  128, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 4, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.n <= 1536) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 3, 4,   64, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 3, 2,  128, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 3, 1,  256, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 6, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.n <= 2048) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 1, 1,  256, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 1,  256, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 4, 1,  256, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 8, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.n <= 3072) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 3, 1,  128, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 3, 1,  256, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 6, 1,  256, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 3, 1, 1024, 1,  true, false>>(s, a);
    }
    else if(a.n <= 4096) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 1,  256, 8,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 4, 1,  256, 4,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 1, 1024, 2,  true, false>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 4, 1, 1024, 1,  true, false>>(s, a);
    }
    else if(a.n > 4096) {
        if (a.n % 8 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 1,  256, 8,  true, true>>(s, a);
        else if (a.n % 4 == 0)
            r = smoothquant_<trait_<data_type,  1, 4, 1,  256, 4,  true, true>>(s, a);
        else if (a.n % 2 == 0)
            r = smoothquant_<trait_<data_type,  1, 2, 1, 1024, 2,  true, true>>(s, a);
        else
            r = smoothquant_<trait_<data_type,  1, 4, 1, 1024, 1,  true, true>>(s, a);
    }
    return r;
    // clang-format on
}

float smoothquant(smoothquant_traits t, smoothquant_args a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        return smoothquant_dispatch<ck_tile::fp16_t>(t, a, s);
    }
    else if(t.data_type.compare("bf16") == 0)
    {
        return smoothquant_dispatch<ck_tile::bf16_t>(t, a, s);
    }
    else
        throw std::runtime_error("Without supported instances!");
}
