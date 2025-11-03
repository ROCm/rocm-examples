// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation of rights
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
#include "example_utils.hpp"

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <fstream>
#include <iostream>
#include <op_normalize.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Normalize operation example.
 */

/**
 * @brief Example shift base parameter file content
 * 1 <-- number of images
 * 1 <-- scalar base indicator for image 1 (1: scalar base; 0: per pixel per channel shift)
 * 120.0 <-- base for channel 0
 * 110.0 <-- base for channel 1
 * 115.0 <-- base for channel 2
 */

/**
 * @brief Example scale parameter file content
 * 1 <-- number of images
 * 1 <-- scalar scale indicator for image 1 (1: scalar scale; 0: per pixel per channel scaling)
 * 80.0 <-- scale for channel 0
 * 75.0 <-- scale for channel 1
 * 65.0 <-- scale for channel 2
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<bool>("cpu", "cpu", false, "Use CPU instead of GPU");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<float>("global_scale", "global_scale", 1.0f, "Global scale parameter");
    parser.set_optional<float>("global_shift", "global_shift", 0.0f, "Global shift parameter");
    parser.set_optional<std::string>("base_file", "base_file", "", "Shifting base parameter file");
    parser.set_optional<std::string>("scale_file", "scale_file", "", "Scaling parameter file");
    parser.set_optional<int>("stddev_scale",
                             "stddev_scale",
                             0,
                             "Scaling parameter is standard deviation (0/1)");
    parser.set_optional<float>("epsilon", "epsilon", 0.1f, "Epsilon parameter");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path  = parser.get<std::string>("i");
    std::string output_file_path = parser.get<std::string>("o");
    bool        use_gpu          = !parser.get<bool>("cpu");
    int         device_id        = parser.get<int>("d");
    float       global_scale     = parser.get<float>("global_scale");
    float       global_shift     = parser.get<float>("global_shift");
    std::string base_file_path   = parser.get<std::string>("base_file");
    std::string scale_file_path  = parser.get<std::string>("scale_file");
    int         stddev_scale     = parser.get<int>("stddev_scale");
    float       epsilon          = parser.get<float>("epsilon");
    bool        base_set         = !base_file_path.empty();
    bool        scale_set        = !scale_file_path.empty();
    uint32_t    flags            = stddev_scale ? ROCCV_NORMALIZE_SCALE_IS_STDDEV : 0;

    // Set up device and stream
    eDeviceType device;
    hipStream_t stream = nullptr;

    if(use_gpu)
    {
        device = eDeviceType::GPU;
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipStreamCreate(&stream));
    }
    else
    {
        device = eDeviceType::CPU;
    }

    // Load input image
    cv::Mat image_data = cv::imread(input_file_path);
    if(image_data.empty())
    {
        std::cerr << "Error: Failed to read the input image file: " << input_file_path << std::endl;
        return error_exit_code;
    }

    // Set up scale tensor
    int                scale_batch_size;
    Size2D             scale_size;
    roccv::ImageFormat scale_format = roccv::FMT_RGBf32;
    std::vector<float> scale_data;

    if(scale_set)
    {
        std::ifstream scale_param_file(scale_file_path);
        if(scale_param_file.is_open())
        {
            std::string line;
            std::getline(scale_param_file, line);
            scale_batch_size = std::stoi(line.c_str());
            if(scale_batch_size > 0)
            {
                for(int i = 0; i < scale_batch_size; i++)
                {
                    std::getline(scale_param_file, line);
                    int scalar_scale = std::stoi(line.c_str());
                    if(scalar_scale)
                    {
                        scale_size   = {1, 1};
                        int curr_idx = scale_data.size();
                        scale_data.resize(curr_idx + 3); // 3 channels
                        for(int b = curr_idx; b < 3 + curr_idx; b++)
                        {
                            std::getline(scale_param_file, line);
                            scale_data[b] = std::atof(line.c_str());
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Per pixel scale is not supported in current sample."
                                  << std::endl;
                        return error_exit_code;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Invalid scale batch size: " << scale_batch_size << std::endl;
                return error_exit_code;
            }
        }
        else
        {
            std::cerr << "Error: Failed to open scale parameter file " << scale_file_path
                      << std::endl;
            return error_exit_code;
        }
    }
    else
    {
        // Use default scale params
        scale_batch_size = 1;
        scale_size       = {1, 1};
        scale_data       = {1.0, 1.0, 1.0};
    }

    Tensor scale_tensor(scale_batch_size, scale_size, scale_format, device);
    auto   scale_tensor_data = scale_tensor.exportData<TensorDataStrided>();

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(scale_tensor_data.basePtr(),
                                 scale_data.data(),
                                 scale_data.size() * sizeof(float),
                                 hipMemcpyHostToDevice,
                                 stream));
    }
    else
    {
        memcpy(scale_tensor_data.basePtr(), scale_data.data(), scale_data.size() * sizeof(float));
    }

    // Set up base tensor
    int                base_batch_size;
    Size2D             base_size;
    roccv::ImageFormat base_format = roccv::FMT_RGBf32;
    std::vector<float> base_data;

    if(base_set)
    {
        std::ifstream base_param_file(base_file_path);
        if(base_param_file.is_open())
        {
            std::string line;
            std::getline(base_param_file, line);
            base_batch_size = std::stoi(line.c_str());
            if(base_batch_size > 0)
            {
                for(int i = 0; i < base_batch_size; i++)
                {
                    std::getline(base_param_file, line);
                    int scalar_base = std::stoi(line.c_str());
                    if(scalar_base)
                    {
                        base_size    = {1, 1};
                        int curr_idx = base_data.size();
                        base_data.resize(curr_idx + 3); // 3 channels
                        for(int b = curr_idx; b < 3 + curr_idx; b++)
                        {
                            std::getline(base_param_file, line);
                            base_data[b] = std::atof(line.c_str());
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Per pixel shift is not supported in current sample."
                                  << std::endl;
                        return error_exit_code;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Invalid base batch size: " << base_batch_size << std::endl;
                return error_exit_code;
            }
        }
        else
        {
            std::cerr << "Error: Failed to open base parameter file " << base_file_path
                      << std::endl;
            return error_exit_code;
        }
    }
    else
    {
        // Use default base params
        base_batch_size = 1;
        base_size       = {1, 1};
        base_data       = {0.0, 0.0, 0.0};
    }

    Tensor base_tensor(base_batch_size, base_size, base_format, device);
    auto   base_tensor_data = base_tensor.exportData<TensorDataStrided>();

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(base_tensor_data.basePtr(),
                                 base_data.data(),
                                 base_data.size() * sizeof(float),
                                 hipMemcpyHostToDevice,
                                 stream));
    }
    else
    {
        memcpy(base_tensor_data.basePtr(), base_data.data(), base_data.size() * sizeof(float));
    }

    // Create input/output tensors for the image.
    TensorShape image_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                            {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType    dtype(eDataType::DATA_TYPE_U8);
    Tensor      input(image_shape, dtype, device);
    Tensor      output(image_shape, dtype, device);

    // Move image data to input tensor
    size_t image_size_in_bytes = input.shape().size() * input.dtype().size();
    auto   input_data          = input.exportData<TensorDataStrided>();

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(input_data.basePtr(),
                                 image_data.data,
                                 image_size_in_bytes,
                                 hipMemcpyHostToDevice,
                                 stream));
    }
    else
    {
        memcpy(input_data.basePtr(), image_data.data, image_size_in_bytes);
    }

    Normalize op;
    op(stream,
       input,
       base_tensor,
       scale_tensor,
       output,
       global_scale,
       global_shift,
       epsilon,
       flags,
       device);

    // Move image data back to host
    size_t               output_size = output.shape().size() * output.dtype().size();
    auto                 out_data    = output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(output_size);

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(h_output.data(),
                                 out_data.basePtr(),
                                 output_size,
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    else
    {
        memcpy(h_output.data(), out_data.basePtr(), output_size);
    }

    // Write output image to disk
    cv::Mat out_image_data(image_data.rows, image_data.cols, image_data.type(), h_output.data());
    bool    ret = cv::imwrite(output_file_path, out_image_data);
    if(!ret)
    {
        std::cerr << "Error: Failed to save output image to file: " << output_file_path
                  << std::endl;
        return error_exit_code;
    }

    // Print results
    std::cout << "Input image file: " << input_file_path << std::endl;
    std::cout << "Output image file: " << output_file_path << std::endl;
    if(use_gpu)
    {
        std::cout << "Operation on GPU device " << device_id << std::endl;
    }
    else
    {
        std::cout << "Operation on CPU" << std::endl;
    }
    std::cout << "Image size: width = " << image_data.cols << ", height = " << image_data.rows
              << std::endl;
    std::cout << "Global scale: " << global_scale << ", Global shift: " << global_shift
              << ", Epsilon: " << epsilon << std::endl;

    // Clean up
    if(use_gpu && stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    return EXIT_SUCCESS;
}
