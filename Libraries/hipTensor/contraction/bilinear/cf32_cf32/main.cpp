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
        return 0;
    }

    // 2. Define data types.
    typedef hipFloatComplex data_type_a;
    typedef hipFloatComplex data_type_b;
    typedef hipFloatComplex data_type_c;
    typedef hipFloatComplex float_type_compute;

    // 3. Set up tensor data types.
    constexpr hiptensorDataType_t          type_a       = HIPTENSOR_C_32F;
    constexpr hiptensorDataType_t          type_b       = HIPTENSOR_C_32F;
    constexpr hiptensorDataType_t          type_c       = HIPTENSOR_C_32F;
    constexpr hiptensorComputeDescriptor_t type_compute = HIPTENSOR_COMPUTE_DESC_C32F;

    // 4. Set scalar values.
    float_type_compute alpha{1.0f, 1.0f};
    float_type_compute beta{1.0f, 1.0f};

    // 5. Run bilinear contraction sample.
    return bilinear_contraction_sample<data_type_a,
                                       data_type_b,
                                       data_type_c,
                                       type_a,
                                       type_b,
                                       type_c,
                                       type_compute>(&alpha, &beta);
}
