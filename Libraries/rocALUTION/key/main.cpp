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

#include "CmdParser/cmdparser.hpp"

#include <iostream>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("matrix",
                                     "matrix",
                                     std::string(EXAMPLE_DATA_DIR) + "/gr_30_30.mtx",
                                     "Path to matrix file in MTX format");
    parser.run_and_exit_if_error();

    std::string matrix_file = parser.get<std::string>("matrix");

    // Initialize rocALUTION
    init_rocalution();

    // rocALUTION objects
    LocalMatrix<double> mat;

    // Read matrix from MTX file
    mat.ReadFileMTX(matrix_file);

    // Print matrix info
    mat.Info();

    long int row_key;
    long int col_key;
    long int val_key;

    // Compute keys
    mat.Key(row_key, col_key, val_key);

    // Print keys
    std::cout << "Row key = " << row_key << std::endl
              << "Col key = " << col_key << std::endl
              << "Val key = " << val_key << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}
