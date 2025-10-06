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
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef EXAMPLE_DATA_DIR
#define EXAMPLE_DATA_DIR "."
#endif

void read_nth_digit(const std::string& digits_file, const int n, std::vector<float>& digit);

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    const std::string default_model_path = std::string(EXAMPLE_DATA_DIR) + "/mnist-8.onnx";
    const std::string default_digits_path = std::string(EXAMPLE_DATA_DIR) + "/digits.txt";
    parser.set_optional<std::string>("model", "model", default_model_path, "Path to ONNX model file");
    parser.set_optional<std::string>("digits", "digits", default_digits_path, "Path to digits data file");
    parser.set_optional<std::string>("target", "target", "ref", "Target device: cpu, gpu, or ref");
    parser.set_optional<bool>("fp16", "fp16", false, "Enable FP16 quantization");
    parser.set_optional<bool>("int8", "int8", false, "Enable INT8 quantization");
    parser.set_optional<bool>("calibration",
                              "calibration",
                              false,
                              "Enable INT8 calibration (requires --int8)");
    parser.set_optional<bool>("print", "print", false, "Print graph at each stage");
    parser.run_and_exit_if_error();

    // Get arguments
    const std::string model_file      = parser.get<std::string>("model");
    const std::string digits_file     = parser.get<std::string>("digits");
    const std::string target_str      = parser.get<std::string>("target");
    const bool        use_fp16        = parser.get<bool>("fp16");
    const bool        use_int8        = parser.get<bool>("int8");
    const bool        use_calibration = parser.get<bool>("calibration");
    const bool        print_graph     = parser.get<bool>("print");

    // Validate target
    if(target_str != "cpu" && target_str != "gpu" && target_str != "ref")
    {
        std::cerr << "Error: Invalid target '" << target_str << "'. Must be cpu, gpu, or ref"
                  << std::endl;
        return error_exit_code;
    }

    // Parse ONNX model
    std::cout << "Parsing ONNX model: " << model_file << std::endl;
    migraphx::program      prog;
    migraphx::onnx_options onnx_opts;
    MIGRAPHX_CHECK(prog = migraphx::parse_onnx(model_file.c_str(), onnx_opts));

    if(print_graph)
    {
        prog.print();
        std::cout << std::endl;
    }

    // Create target
    migraphx::target targ = migraphx::target(target_str.c_str());

    // Apply quantization if requested
    if(use_fp16)
    {
        std::cout << "Quantizing program for FP16..." << std::endl;
        MIGRAPHX_CHECK(migraphx::quantize_fp16(prog));

        if(print_graph)
        {
            prog.print();
            std::cout << std::endl;
        }
    }
    else if(use_int8)
    {
        std::cout << "Quantizing program for INT8..." << std::endl;

        if(use_calibration)
        {
            std::cout << "Using calibration data" << std::endl;
            std::vector<float> calib_dig;
            read_nth_digit(digits_file, 9, calib_dig);

            migraphx::quantize_int8_options quant_opts;
            migraphx::program_parameters    quant_params;
            auto                            param_shapes = prog.get_parameter_shapes();
            for(auto&& name : param_shapes.names())
            {
                quant_params.add(name, migraphx::argument(param_shapes[name], calib_dig.data()));
            }

            quant_opts.add_calibration_data(quant_params);
            MIGRAPHX_CHECK(migraphx::quantize_int8(prog, targ, quant_opts));
        }
        else
        {
            MIGRAPHX_CHECK(migraphx::quantize_int8(prog, targ, migraphx::quantize_int8_options()));
        }

        if(print_graph)
        {
            prog.print();
            std::cout << std::endl;
        }
    }

    // Compile program
    std::cout << "Compiling program for " << target_str << "..." << std::endl;
    if(target_str == "gpu")
    {
        migraphx::compile_options comp_opts;
        comp_opts.set_offload_copy();
        MIGRAPHX_CHECK(prog.compile(targ, comp_opts));
    }
    else
    {
        MIGRAPHX_CHECK(prog.compile(targ));
    }

    if(print_graph)
    {
        prog.print();
        std::cout << std::endl;
    }

    // Prepare input data
    std::vector<float>                 digit;
    std::random_device                 rd;
    std::uniform_int_distribution<int> dist(0, 9);
    const int                          rand_digit = dist(rd);
    std::cout << "Model input:" << std::endl;
    read_nth_digit(digits_file, rand_digit, digit);

    migraphx::program_parameters prog_params;
    auto                         param_shapes = prog.get_parameter_shapes();
    auto                         input        = param_shapes.names().front();
    prog_params.add(input, migraphx::argument(param_shapes[input], digit.data()));

    // Execute program
    std::cout << "Model evaluating input..." << std::endl;
    auto start   = std::chrono::high_resolution_clock::now();
    auto outputs = prog.eval(prog_params);
    auto stop    = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Inference complete" << std::endl;
    std::cout << "Inference time: " << elapsed.count() * 1e-3 << "ms" << std::endl;

    // Process results
    auto shape   = outputs[0].get_shape();
    auto lengths = shape.lengths();
    auto num_results
        = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>());
    float*       results = reinterpret_cast<float*>(outputs[0].data());
    const float* max     = std::max_element(results, results + num_results);
    int          answer  = max - results;

    std::cout << std::endl
              << "Randomly chosen digit: " << rand_digit << std::endl
              << "Result from inference: " << answer << std::endl
              << std::endl
              << (answer == rand_digit ? "CORRECT" : "INCORRECT") << std::endl
              << std::endl;

    return (answer == rand_digit) ? 0 : error_exit_code;
}

void read_nth_digit(const std::string& digits_file, const int n, std::vector<float>& digit)
{
    const std::string symbols = "@0#%=+*-.  ";
    std::ifstream     file(digits_file);
    const int         digits_count = 10;
    const int         height       = 28;
    const int         width        = 28;

    if(!file.is_open())
    {
        std::cerr << "Error: Could not open digits file: " << digits_file << std::endl;
        std::exit(error_exit_code);
    }

    for(int d = 0; d < digits_count; ++d)
    {
        for(int i = 0; i < height * width; ++i)
        {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            if(d == n)
            {
                float data = temp / 255.0f;
                digit.push_back(data);
                std::cout << symbols[static_cast<int>(data * 10) % 11];
                if((i + 1) % width == 0)
                {
                    std::cout << std::endl;
                }
            }
        }
    }
    std::cout << std::endl;
}
