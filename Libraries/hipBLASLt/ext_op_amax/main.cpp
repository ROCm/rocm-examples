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

#include "example_utils.hpp"
#include "hipblaslt_utils.hpp"

#include <hipblaslt/hipblaslt-ext-op.h>

void amax(hipDataType type,
          hipDataType dtype,
          void*       d_out,
          void*       d_in,
          int64_t     m,
          int64_t     n,
          hipStream_t stream);

int main()
{
    /** This is a amax example
     *  in  = (m, n). lda = m
     *  out = (1). ldb = 1
     */
    opt_amax_runner<float> runner_f32(135, 345);

    runner_f32.run(
        [&runner_f32]
        {
            amax(HIP_R_32F,
                 HIP_R_32F,
                 runner_f32.d_out,
                 runner_f32.d_in,
                 runner_f32.m,
                 runner_f32.n,
                 runner_f32.stream);
        });

    opt_amax_runner<hipblasLtHalf> runner_f16(135, 345);

    runner_f16.run(
        [&runner_f16]
        {
            amax(HIP_R_16F,
                 HIP_R_16F,
                 runner_f16.d_out,
                 runner_f16.d_in,
                 runner_f16.m,
                 runner_f16.n,
                 runner_f16.stream);
        });

    return 0;
}

void amax(hipDataType type,
          hipDataType dtype,
          void*       d_out,
          void*       d_in,
          int64_t     m,
          int64_t     n,
          hipStream_t stream)
{
    HIPBLASLT_CHECK(hipblasltExtAMax(type, dtype, d_out, d_in, m, n, stream));
}
