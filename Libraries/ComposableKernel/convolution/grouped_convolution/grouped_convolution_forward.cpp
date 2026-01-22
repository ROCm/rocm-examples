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

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"
#include "grouped_convolution_forward_invoker.hpp"
#include "run_grouped_convolution_fwd_example.inc"

template <template <typename PrecType> typename ConvConfig>
int run_grouped_conv_fwd_example(int argc, char* argv[])
{
    using Invoker = GroupedConvolutionForwardInvoker;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::half_t>,
                                                      ck_tile::half_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::bf16_t>,
                                                      ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
#if CK_TILE_USE_WMMA
    return !run_grouped_conv_fwd_example<ConvConfigComputeV3_WMMA>(argc, argv);
#else
    return !run_grouped_conv_fwd_example<ConvConfigComputeV3>(argc, argv);
#endif
}
