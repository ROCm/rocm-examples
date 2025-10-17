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


#include <ck_tile/core/arch/arch.hpp>

#include <stdexcept>
#include <string>
#include <variant>

auto string_to_datatype(const std::string& datatype)
{
    using PrecVariant = std::variant<ck_tile::half_t, ck_tile::bf16_t, float>;

    if(datatype == "fp16")
    {
        return PrecVariant{ck_tile::half_t{}};
    }
    else if(datatype == "bf16")
    {
        return PrecVariant{ck_tile::bf16_t{}};
    }
    else if(datatype == "fp32")
    {
        return PrecVariant{float{}};
    }
    else
    {
        throw std::runtime_error("Unsupported data type: " + datatype);
    }
};
