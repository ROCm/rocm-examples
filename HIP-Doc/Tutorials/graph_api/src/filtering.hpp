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

#ifndef FILTERING_HPP
#define FILTERING_HPP

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <cstddef>
#include <cstdint>

__global__ void filter_creation_kernel(float* __restrict__ r, std::int32_t N_hFFT, float tau);

__global__ void filter_absolute_kernel(hipfftComplex* R, std::uint32_t N_hTrans, float tau);

__global__ void filter_application_kernel(hipfftComplex* __restrict__ P, std::size_t const pitch,
                                          hipfftComplex const* __restrict__ R,
                                          std::uint32_t N_hTrans, std::uint32_t N_v);

__global__ void filter_normalization_kernel(float* p, std::size_t const pitch,
                                            std::uint32_t const N_hFFT,
                                            std::uint32_t const N_h, std::uint32_t const N_v);

#endif
