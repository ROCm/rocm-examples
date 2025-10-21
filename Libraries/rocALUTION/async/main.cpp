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

#include <cstdlib>
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
    parser.set_optional<int>("threads", "threads", 0, "Number of OMP threads (0 = default)");
    parser.run_and_exit_if_error();

    std::string matrix_file = parser.get<std::string>("matrix");
    int         num_threads = parser.get<int>("threads");

    // Initialize rocALUTION
    init_rocalution();

    // Set number of OMP threads if specified
    if(num_threads > 0)
    {
        set_omp_threads_rocalution(num_threads);
    }

    // Print rocALUTION info
    info_rocalution();

    // rocALUTION objects
    LocalVector<double> x;
    LocalVector<double> y;
    LocalMatrix<double> mat;

    double tick, tack;
    double tickg, tackg;

    // Read matrix from MTX file
    mat.ReadFileMTX(matrix_file);

    // Allocate vectors
    x.Allocate("x", mat.GetN());
    y.Allocate("y", mat.GetM());

    // Initialize x with 1
    x.Ones();

    // No Async
    tickg = rocalution_time();

    // Initialize y with 0
    y.Zeros();

    // Print object info
    mat.Info();
    x.Info();
    y.Info();

    // CPU
    tick = rocalution_time();

    // y += mat * x
    for(int i = 0; i < 100; ++i)
    {
        mat.ApplyAdd(x, 1.0, &y);
    }

    tack = rocalution_time();
    std::cout << "CPU execution took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    std::cout << "Dot product = " << x.Dot(y) << std::endl;

    tick = rocalution_time();

    // Memory transfer
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    y.MoveToAccelerator();

    // Print object info
    mat.Info();
    x.Info();
    y.Info();

    tack = rocalution_time();
    std::cout << "Sync transfer took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Initialize y with 0
    y.Zeros();

    // Accelerator
    tick = rocalution_time();

    // y += mat * x
    for(int i = 0; i < 100; ++i)
    {
        mat.ApplyAdd(x, 1.0, &y);
    }

    tack = rocalution_time();
    std::cout << "Accelerator execution took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    std::cout << "Dot product = " << x.Dot(y) << std::endl;

    tackg = rocalution_time();
    std::cout << "Total execution + transfers (no async) took: " << (tackg - tickg) / 1e6 << " sec"
              << std::endl;

    // Move data to host
    mat.MoveToHost();
    x.MoveToHost();
    y.MoveToHost();

    // Initialize y with 0
    y.Zeros();

    // Async
    tickg = rocalution_time();
    tick  = rocalution_time();

    // Memory transfer
    mat.MoveToAcceleratorAsync();
    x.MoveToAcceleratorAsync();

    // Print oject info
    mat.Info();
    x.Info();
    y.Info();

    tack = rocalution_time();
    std::cout << "Async transfer took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // CPU
    tick = rocalution_time();

    // y += mat * x
    for(int i = 0; i < 100; ++i)
    {
        mat.ApplyAdd(x, 1.0, &y);
    }

    tack = rocalution_time();
    std::cout << "CPU execution took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    std::cout << "Dot product = " << x.Dot(y) << std::endl;

    // Synchronize objects
    mat.Sync();
    x.Sync();

    // Move y to host
    y.MoveToAccelerator();

    // Print object info
    mat.Info();
    x.Info();
    y.Info();

    // Initialize y with 0
    y.Zeros();

    // Accelerator
    tick = rocalution_time();

    // y += mat * x
    for(int i = 0; i < 100; ++i)
    {
        mat.ApplyAdd(x, 1.0, &y);
    }

    tack = rocalution_time();
    std::cout << "Accelerator execution took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    std::cout << "Dot product = " << x.Dot(y) << std::endl;

    tackg = rocalution_time();
    std::cout << "Total execution + transfers (async) took: " << (tackg - tickg) / 1e6 << " sec"
              << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}
