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

#include "hiptensor_utils.hpp"

int main()
{
    // 1. Check if F32 is supported.
    if(!is_f32_supported())
    {
        std::cout << "unsupported host device" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 2. Define data types.
    typedef float data_type_a;
    typedef float data_type_b;
    typedef float data_type_d;
    typedef float float_type_compute;

    // 3. Set up tensor data types.
    constexpr hiptensorDataType_t          type_a       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          type_b       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          type_d       = HIPTENSOR_R_32F;
    constexpr hiptensorComputeDescriptor_t type_compute = HIPTENSOR_COMPUTE_DESC_32F;

    // 4. Set scalar values.
    float_type_compute alpha = 1;

    // 5. Run scale contraction sample.
    return scale_contraction_sample<data_type_a,
                                    data_type_b,
                                    data_type_d,
                                    type_a,
                                    type_b,
                                    type_d,
                                    type_compute>(&alpha);
}
