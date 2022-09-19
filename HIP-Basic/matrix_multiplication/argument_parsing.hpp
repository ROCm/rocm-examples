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

#ifndef HIP_BASIC_MATRIX_MULTIPLICATION_ARGUMENT_PARSING_HPP
#define HIP_BASIC_MATRIX_MULTIPLICATION_ARGUMENT_PARSING_HPP

#include "example_utils.hpp"

#include <charconv>
#include <iostream>
#include <string>
#include <string_view>

#include <cstdlib>

/// \brief Tries to read the matrix dimensions from the command line.
/// If no command line arguments were provided, the passed values are not modified.
/// Otherwise, the number of arguments must be 3: <A rows> <A columns> <B columns>
/// (B rows will be equal to A columns).
/// If the number of arguments is different, or the arguments cannot be parsed to
/// unsigned ints, an error message is printed and the program exits with a non-zero code.
inline void matrix_dimensions_from_command_line(const int          argc,
                                                const char*        argv[],
                                                unsigned int&      a_rows,
                                                unsigned int&      a_cols,
                                                unsigned int&      b_cols,
                                                const unsigned int block_size)
{
    const auto print_usage_and_exit = [=]()
    {
        const std::string usage_message
            = "Calculates matrix product A*B.\n"
              "Usage: hip_matrix_multiplication [<A rows> <A columns> <B columns>].\n"
              "Matrix dimensions must be positive multiples of block_size ("
              + std::to_string(block_size) + ")";
        std::cout << usage_message << std::endl;
        exit(error_exit_code);
    };
    const auto get_argument_by_index = [=](const unsigned int index) -> unsigned int
    {
        const std::string_view argument_text(argv[index]);

        unsigned int converted_value;
        const auto   conversion_result = std::from_chars(argument_text.data(),
                                                       argument_text.data() + argument_text.size(),
                                                       converted_value);
        if(conversion_result.ec != std::errc{} || (converted_value % block_size) != 0)
        {
            print_usage_and_exit();
        }
        return converted_value;
    };

    if(argc == 1)
    {
        return;
    }
    if(argc != 4)
    {
        print_usage_and_exit();
    }
    a_rows = get_argument_by_index(1);
    a_cols = get_argument_by_index(2);
    b_cols = get_argument_by_index(3);
}

#endif // HIP_BASIC_MATRIX_MULTIPLICATION_ARGUMENT_PARSING_HPP
