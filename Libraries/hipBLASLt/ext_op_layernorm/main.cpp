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

void layer_norm(hipDataType type,
                void*       d_out,
                void*       d_mean,
                void*       d_invvar,
                void*       d_in,
                int64_t     m,
                int64_t     n,
                float       eps,
                void*       d_gamma,
                void*       d_beta,
                hipStream_t stream);

int main()
{
    layer_norm_runner<float> runner_f32(135, 345);

    runner_f32.run(
        [&runner_f32]
        {
            layer_norm(HIP_R_32F,
                       runner_f32.d_out,
                       runner_f32.d_mean,
                       runner_f32.d_invvar,
                       runner_f32.d_in,
                       runner_f32.m,
                       runner_f32.n,
                       1e-5,
                       runner_f32.d_gamma,
                       runner_f32.d_beta,
                       runner_f32.stream);
        });

    return 0;
}

void layer_norm(hipDataType type,
                void*       d_out,
                void*       d_mean,
                void*       d_invvar,
                void*       d_in,
                int64_t     m,
                int64_t     n,
                float       eps,
                void*       d_gamma,
                void*       d_beta,
                hipStream_t stream)
{
    HIPBLASLT_CHECK(hipblasltExtLayerNorm(type,
                                          d_out,
                                          d_mean,
                                          d_invvar,
                                          d_in,
                                          m,
                                          n,
                                          eps,
                                          d_gamma,
                                          d_beta,
                                          stream));
}
