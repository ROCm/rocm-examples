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
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fused_moe.hpp"

struct moe_sorting_trait
{
    std::string index_type;
    std::string weight_type;         // currently always float
    bool local_expert_masking;       // if mask experts as local expert
    bool clear_workspace_inside_api; // if true, no need clear workspace outsize (will take care of
                                     // it inside API)
    int dispatch_policy; // 0 - let the API choose kernel for you. 1 - always use single kerenl. 2 -
                         // always use mp kernel NOTE: moe_sorting_get_workspace_size() need use
                         // same dispatch_policy value. it will be undefined behavior if ppl using
                         // different value when get ws and call the kernel
};

struct moe_sorting_args : public ck_tile::MoeSortingHostArgs
{
};

// use below API before call moe_sorting() to indicate if need workspace or not
// if return non zero, means need workspace, you need to allocate a GPU buffer
// and set to moe_sorting_args.p_ws
// NOTE: workspace size are required to clear zero before use the API
int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk, int dispatch_policy);
float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
float moe_sorting_mp(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
