// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_EXAMPLE_UTILS_HPP
#define COMMON_EXAMPLE_UTILS_HPP

#include <cassert>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <hip/hip_runtime.h>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

/// \brief Formats a range of elements to a pretty string.
/// \tparam BidirectionalIterator - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to
/// \p std::ostream.
template<class BidirectionalIterator>
std::string format_range(const BidirectionalIterator begin, const BidirectionalIterator end)
{
    std::stringstream sstream;
    sstream << "[ ";
    for(auto it = begin; it != end; ++it)
    {
        sstream << *it;
        if(it != std::prev(end))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

/// \brief Formats a range of pairs to a pretty string. The length of the two ranges must match.
/// \tparam BidirectionalIteratorT - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
/// \tparam BidirectionalIteratorU - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
template<class BidirectionalIteratorT, typename BidirectionalIteratorU>
std::string format_pairs(const BidirectionalIteratorT begin_a,
                         const BidirectionalIteratorT end_a,
                         const BidirectionalIteratorU begin_b,
                         const BidirectionalIteratorU end_b)
{
    (void)end_b;
    assert(std::distance(begin_a, end_a) == std::distance(begin_b, end_b));

    std::stringstream sstream;
    sstream << "[ ";
    auto it_a = begin_a;
    auto it_b = begin_b;
    for(; it_a < end_a; ++it_a, ++it_b)
    {
        sstream << "(" << *it_a << ", " << *it_b << ")";

        if(it_a != std::prev(end_a))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

#endif // COMMON_EXAMPLE_UTILS_HPP
