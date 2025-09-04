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

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <source_location>
#include <stdexcept>
#include <string>

inline void hip_check(hipError_t err, std::source_location const loc = std::source_location::current()) noexcept(false)
{
    if(err != hipSuccess)
    {
        auto msg = std::string{"HIP error at "};
        msg += loc.file_name();
        msg += '(';
        msg += std::to_string(loc.line());
        msg += ':';
        msg += std::to_string(loc.column());
        msg += ") `";
        msg += loc.function_name();
        msg += "`: ";
        msg += hipGetErrorString(err);

        throw std::runtime_error{msg};
    }
}

inline void hipfft_check(hipfftResult res, std::source_location const loc = std::source_location::current()) noexcept(false)
{
    using namespace std::literals::string_literals;
    auto getHipFFTResultString = [](hipfftResult r)
    {
        switch(r)
        {
        case HIPFFT_SUCCESS:
            return "HIPFFT_SUCCESS"s;

        case HIPFFT_INVALID_PLAN:
            return "HIPFFT_INVALID_PLAN"s;

        case HIPFFT_ALLOC_FAILED:
            return "HIPFFT_ALLOC_FAILED"s;

        case HIPFFT_INVALID_VALUE:
            return "HIPFFT_INVALID_VALUE"s;

        case HIPFFT_INTERNAL_ERROR:
            return "HIPFFT_INTERNAL_ERROR"s;

        case HIPFFT_EXEC_FAILED:
            return "HIPFFT_EXEC_FAILED"s;

        case HIPFFT_SETUP_FAILED:
            return "HIPFFT_SETUP_FAILED"s;

        case HIPFFT_INVALID_SIZE:
            return "HIPFFT_INVALID_SIZE"s;

        case HIPFFT_INCOMPLETE_PARAMETER_LIST:
            return "HIPFFT_INCOMPLETE_PARAMETER_LIST"s;

        case HIPFFT_INVALID_DEVICE:
            return "HIPFFT_INVALID_DEVICE"s;

        case HIPFFT_NO_WORKSPACE:
            return "HIPFFT_NO_WORKSPACE"s;

        case HIPFFT_NOT_IMPLEMENTED:
            return "HIPFFT_NOT_IMPLEMENTED"s;

        case HIPFFT_NOT_SUPPORTED:
            return "HIPFFT_NOT_SUPPORTED"s;

        default:
            return "Unknown error."s;
        }

    };

    if(res != HIPFFT_SUCCESS)
    {
        auto msg = std::string{"hipFFT error at "};
        msg += loc.file_name();
        msg += '(';
        msg += std::to_string(loc.line());
        msg += ':';
        msg += std::to_string(loc.column());
        msg += ") `";
        msg += loc.function_name();
        msg += "`: ";
        msg += getHipFFTResultString(res);

        throw std::runtime_error{msg};
    }
}

#endif
