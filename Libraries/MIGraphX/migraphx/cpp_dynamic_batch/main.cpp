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
#include "migraphx_utils.hpp"

#include <migraphx/migraphx.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#ifndef EXAMPLE_DATA_DIR
#define EXAMPLE_DATA_DIR "."
#endif

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    const std::string default_onnx_path = std::string(EXAMPLE_DATA_DIR) + "/add_scalar_test.onnx";
    parser.set_optional<std::string>("onnx",
                                     "onnx",
                                     default_onnx_path,
                                     "Path to ONNX model file");
    parser.set_optional<int>("batch", "batch", 2, "Batch size for dynamic batch processing");
    parser.run_and_exit_if_error();

    // Get arguments
    const std::string onnx_file  = parser.get<std::string>("onnx");
    const int         batch_size = parser.get<int>("batch");

    if(batch_size <= 0)
    {
        std::cerr << "Error: Batch size must be positive" << std::endl;
        return error_exit_code;
    }

    // Set up dynamic dimensions
    migraphx::onnx_options       o_options;
    migraphx::dynamic_dimensions dyn_dims = {
        migraphx::dynamic_dimension{1, 4, {2, 4}},
        migraphx::dynamic_dimension{3, 3},
        migraphx::dynamic_dimension{4, 4},
        migraphx::dynamic_dimension{5, 5}
    };
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);

    // Parse ONNX model
    migraphx::program p;
    MIGRAPHX_CHECK(p = migraphx::parse_onnx(onnx_file.c_str(), o_options));

    // Compile program
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    MIGRAPHX_CHECK(p.compile(migraphx::target("gpu"), c_options));

    // Prepare input data
    const int            total_elements = batch_size * 3 * 4 * 5;
    std::vector<uint8_t> a(total_elements, 3);
    std::vector<uint8_t> b = {2};

    // Set up program parameters
    migraphx::program_parameters pp;
    migraphx::shape              s
        = migraphx::shape(migraphx_shape_uint8_type, {static_cast<size_t>(batch_size), 3, 4, 5});
    pp.add("0", migraphx::argument(s, a.data()));
    pp.add("1", migraphx::argument(migraphx::shape(migraphx_shape_uint8_type, {1}, {0}), b.data()));

    // Execute program
    auto outputs = p.eval(pp);
    auto result  = outputs[0];

    // Verify result
    std::vector<uint8_t> c(total_elements, 5);
    if(bool{result == migraphx::argument(s, c.data())})
    {
        std::cout << "Successfully executed dynamic batch add with batch size " << batch_size
                  << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Failed dynamic batch add" << std::endl;
        return error_exit_code;
    }
}
