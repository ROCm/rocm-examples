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
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    const std::string default_onnx_path = std::string(EXAMPLE_DATA_DIR) + "/add_scalar_test.onnx";
    parser.set_optional<std::string>("input",
                                     "input",
                                     default_onnx_path,
                                     "Path to input file (ONNX or MIGraphX)");
    parser.set_optional<std::string>("format",
                                     "format",
                                     "onnx",
                                     "Input format: onnx, json, or msgpack");
    parser.set_optional<std::string>("output",
                                     "output",
                                     "",
                                     "Path to output file (saves as msgpack if provided)");
    parser.set_optional<bool>("print", "print", false, "Print the program graph");
    parser.run_and_exit_if_error();

    // Get arguments
    const std::string input_file  = parser.get<std::string>("input");
    const std::string format      = parser.get<std::string>("format");
    const std::string output_file = parser.get<std::string>("output");
    const bool        print_graph = parser.get<bool>("print");

    migraphx::program p;

    // Load or parse the input file based on format
    if(format == "onnx")
    {
        std::cout << "Parsing ONNX file: " << input_file << std::endl;
        migraphx::onnx_options options;
        MIGRAPHX_CHECK(p = migraphx::parse_onnx(input_file.c_str(), options));
    }
    else if(format == "json")
    {
        std::cout << "Loading JSON file: " << input_file << std::endl;
        migraphx::file_options options;
        options.set_file_format("json");
        MIGRAPHX_CHECK(p = migraphx::load(input_file.c_str(), options));
    }
    else if(format == "msgpack")
    {
        std::cout << "Loading msgpack file: " << input_file << std::endl;
        migraphx::file_options options;
        options.set_file_format("msgpack");
        MIGRAPHX_CHECK(p = migraphx::load(input_file.c_str(), options));
    }
    else
    {
        std::cerr << "Error: Unknown format '" << format << "'. "
                  << "Supported formats: onnx, json, msgpack" << std::endl;
        return error_exit_code;
    }

    // Print the program if requested
    if(print_graph)
    {
        std::cout << "\nInput Graph:" << std::endl;
        p.print();
        std::cout << std::endl;
    }

    // Save the program if output file is specified
    if(!output_file.empty())
    {
        std::cout << "Saving program to: " << output_file << std::endl;
        migraphx::file_options options;
        options.set_file_format("msgpack");
        MIGRAPHX_CHECK(migraphx::save(p, output_file.c_str(), options));
        std::cout << "Program saved successfully" << std::endl;
    }

    return 0;
}
