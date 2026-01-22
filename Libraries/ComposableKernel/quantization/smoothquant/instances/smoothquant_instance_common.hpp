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

#include <ck_tile/core.hpp>
#include "smoothquant.hpp"
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = smoothquant_args;

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

template <typename Traits_>
float smoothquant_(const S& s, A a)
{
    using DataType = typename Traits_::DataType;

    using PipelineProblem = ck_tile::SmoothquantPipelineProblem<
        typename SmoothquantTypeConfig<DataType>::XDataType,
        typename SmoothquantTypeConfig<DataType>::SmoothScaleDataType,
        typename SmoothquantTypeConfig<DataType>::ComputeDataType,
        typename SmoothquantTypeConfig<DataType>::YScaleDataType,
        typename SmoothquantTypeConfig<DataType>::QYDataType,
        typename Traits_::Shape,
        Traits_::kPadN,
        Traits_::kTwoPass>;

    using OnePassPipeline = ck_tile::SmoothquantPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::SmoothquantPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using Kernel = ck_tile::Smoothquant<Pipeline>;

    const dim3 grids                       = Kernel::GridSize(a);
    const dim3 blocks                      = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
