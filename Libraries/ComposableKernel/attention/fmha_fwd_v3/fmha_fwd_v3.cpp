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

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

namespace ck_tile {

std::ostream& operator<<(std::ostream& stream, const fmha_fwd_v3_args::data_type_enum& data_type)
{
    switch(data_type)
    {
    case fmha_fwd_v3_args::data_type_enum::fp16: return stream << "fp16";
    case fmha_fwd_v3_args::data_type_enum::bf16: return stream << "bf16";
    default: return stream << "unknown";
    }
}

std::pair<bool, float> fmha_fwd_v3(const fmha_fwd_v3_args& args, const stream_config& config)
{
    if(args.data_type == fmha_fwd_v3_args::data_type_enum::fp16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, false>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, true>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
    }
    else if(args.data_type == fmha_fwd_v3_args::data_type_enum::bf16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, false>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
    }

    return std::make_pair(false, -1.f);
}

} // namespace ck_tile
