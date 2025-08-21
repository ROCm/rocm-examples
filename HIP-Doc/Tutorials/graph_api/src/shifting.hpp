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

#ifndef SHIFTING_HPP
#define SHIFTING_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/*
 Takes the input image and shifts it horizontally and/or vertically by the specified amount of pixels. (0, 0) is in the
 upper left corner. Negative values therefore indicate a left shift (x axis) or a top shift (y axis), positive values
 a right shift (x axis) or bottom shift (y axis).
 */
__global__ void shifting_kernel(std::uint16_t const* __restrict__ in, std::size_t const in_pitch,
                                std::uint16_t* __restrict__ out, std::size_t const out_pitch,
                                std::uint32_t const N_h, std::uint32_t const N_v,
                                std::int32_t const shift_h, std::int32_t const shift_v);

#endif
