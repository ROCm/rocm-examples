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

#include "matrix_core_swizzle.hpp"
#include "matrix_core_swizzle_kernel.hpp"

float matrix_core_swizzle(matrix_core_swizzle_traits t,
                          matrix_core_swizzle_args a,
                          const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        if(t.inst.compare("32x32x8") == 0)
        {
            constexpr int BLOCK_SIZE             = 256;
            constexpr int NPerBlock              = 256;
            constexpr int KPerBlock              = 128;
            constexpr matrix_core_inst_enum Inst = matrix_core_inst_enum::MFMA_32x32x8_F16;
            if(t.permute.compare("0,1,4,2,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,2,4,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,3,4,2,5") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::b_nr_kr_kw_nw_kv;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
        }
        else if(t.inst.compare("16x16x16") == 0)
        {
            constexpr int BLOCK_SIZE             = 256;
            constexpr int NPerBlock              = 256;
            constexpr int KPerBlock              = 128;
            constexpr matrix_core_inst_enum Inst = matrix_core_inst_enum::MFMA_16x16x16_F16;
            if(t.permute.compare("0,1,4,2,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,2,4,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,3,4,2,5") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::b_nr_kr_kw_nw_kv;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
        }
    }
    return -1;
}
