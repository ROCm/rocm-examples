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

#include "fused_moegemm_api_traits.hpp"

#include <ck_tile/ops/fused_moe.hpp>

#include <hip/hip_runtime.h>

#include <iostream>
#include <ostream>

template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

// do not the define of this tepmlate function inside the _api.cpp, otherwise will block make -j
template <typename Ts_>
float fused_moegemm_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly, Ts_::FusedQuant == 1, 1 /*atomic*/>;
    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::FastGeluAsm{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;

    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmEx<f_problem>;
    using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}
