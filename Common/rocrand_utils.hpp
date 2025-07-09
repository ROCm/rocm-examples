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

#ifndef COMMON_ROCRAND_UTILS_HPP
#define COMMON_ROCRAND_UTILS_HPP

#include "example_utils.hpp"

#include <rocrand/rocrand.h>

#include <cstdlib>
#include <iostream>

/// \brief Checks if the provided error code is \p ROCRAND_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCRAND_CHECK(condition)                                                        \
    {                                                                                   \
        const rocrand_status status = condition;                                        \
        if(status != ROCRAND_STATUS_SUCCESS)                                            \
        {                                                                               \
            std::cerr << "rocRAND error encountered at " << __FILE__ << ':' << __LINE__ \
                      << std::endl;                                                     \
            std::exit(error_exit_code);                                                 \
        }                                                                               \
    }

/// \brief Checks whether the values in \p output are uniformly distributed.
/// \return 0 if they are, the mean value obtained otherwise.
double is_not_uniform_dist(std::vector<unsigned int>& output)
{
    double           mean = 0;
    constexpr double tol  = 0.1;

    // Compute mean normalized to [0, 1], as values are generated in [0, 2**32 - 1] = [0, UINT_MAX].
    for(auto v : output)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output.size();

    return (std::abs(mean - 0.5) > tol) ? mean : 0;
}

#endif // COMMON_ROCRAND_UTILS_HPP
