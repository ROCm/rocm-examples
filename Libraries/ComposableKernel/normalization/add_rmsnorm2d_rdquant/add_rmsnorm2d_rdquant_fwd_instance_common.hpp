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

#pragma once

#include "add_rmsnorm2d_rdquant_fwd.hpp"

#include <ck_tile/core.hpp>

#include <hip/hip_runtime.h>

#include <iostream>
#include <ostream>

using S = ck_tile::stream_config;
using A = add_rmsnorm2d_rdquant_fwd_args;

template <typename InputDataType_,
          typename QuantizedDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveInvRms_,
          bool kTwoPass_>
using trait_ = add_rmsnorm2d_rdquant_fwd_traits_<InputDataType_,
                                                 QuantizedDataType_,
                                                 Repeat_M_,
                                                 Repeat_N_,
                                                 ThreadPerBlock_M_,
                                                 ThreadPerBlock_N_,
                                                 Vector_N_,
                                                 kPadN_,
                                                 kSaveInvRms_,
                                                 kTwoPass_>;

template <typename Traits_>
float add_rmsnorm2d_rdquant_fwd_(const S& s, A a)
{
    using InputDataType     = typename Traits_::InputDataType;
    using QuantizedDataType = typename Traits_::QuantizedDataType;

    using PipelineProblem = ck_tile::AddRmsnorm2dRdquantFwdPipelineProblem<
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::ADataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::BDataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::GammaDataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::ComputeDataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::XDataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::YScaleDataType,
        typename AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>::QYDataType,
        typename Traits_::Shape,
        Traits_::kPadN,
        Traits_::kSaveX,
        Traits_::kThreePass>;

    using OnePassPipeline   = ck_tile::AddRmsnorm2dRdquantFwdPipelineOnePass<PipelineProblem>;
    using ThreePassPipeline = ck_tile::AddRmsnorm2dRdquantFwdPipelineThreePass<PipelineProblem>;
    using Pipeline = std::conditional_t<Traits_::kThreePass, ThreePassPipeline, OnePassPipeline>;

    using Kernel = ck_tile::AddRmsnorm2dRdquantFwd<Pipeline>;

    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
