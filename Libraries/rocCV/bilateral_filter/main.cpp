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
#include "example_utils.hpp"

#include <core/tensor.hpp>
#include <fstream>
#include <iostream>
#include <op_bilateral_filter.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<bool>("cpu", "cpu", false, "Use CPU instead of GPU");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<int>("diameter", "diameter", 2, "Diameter of filtering area");
    parser.set_optional<float>("sigma_space", "sigma_space", 2.0f, "Spatial sigma parameter");
    parser.set_optional<float>("sigma_color", "sigma_color", 10.0f, "Color sigma parameter");
    parser.set_optional<int>("border_mode",
                             "border_mode",
                             1,
                             "Border mode (0:constant, 1:replicate, 2:reflect, 3:wrap)");
    parser.set_optional<float>("border_r", "border_r", 0.0f, "Border color red component");
    parser.set_optional<float>("border_g", "border_g", 0.0f, "Border color green component");
    parser.set_optional<float>("border_b", "border_b", 0.0f, "Border color blue component");
    parser.set_optional<float>("border_a", "border_a", 0.0f, "Border color alpha component");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path   = parser.get<std::string>("i");
    std::string output_file_path  = parser.get<std::string>("o");
    bool        use_gpu           = !parser.get<bool>("cpu");
    int         device_id         = parser.get<int>("d");
    int         diameter          = parser.get<int>("diameter");
    float       sigma_space       = parser.get<float>("sigma_space");
    float       sigma_color       = parser.get<float>("sigma_color");
    int         border_mode_value = parser.get<int>("border_mode");

    // Validate border mode
    if(border_mode_value < 0 || border_mode_value > 3)
    {
        std::cerr << "Error: Invalid border mode. Must be 0, 1, 2, or 3." << std::endl;
        return error_exit_code;
    }
    eBorderType border_mode = static_cast<eBorderType>(border_mode_value);

    // Set up border color
    float4 border_color;
    border_color.x = parser.get<float>("border_r");
    border_color.y = parser.get<float>("border_g");
    border_color.z = parser.get<float>("border_b");
    border_color.w = parser.get<float>("border_a");

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

    // Create input/output tensors
    TensorShape image_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                            {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType    dtype(eDataType::DATA_TYPE_U8);
    Tensor      input_tensor(image_shape, dtype, device);
    Tensor      output_tensor(image_shape, dtype, device);

    // Copy image data to device
    size_t image_size_in_bytes = input_tensor.shape().size() * input_tensor.dtype().size();
    auto   input_data          = input_tensor.exportData<TensorDataStrided>();

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

    // Apply bilateral filter
    BilateralFilter bilateral_filter_op;
    bilateral_filter_op(stream,
                        input_tensor,
                        output_tensor,
                        diameter,
                        sigma_color,
                        sigma_space,
                        border_mode,
                        border_color,
                        device);

    // Copy result back to host
    size_t               output_size = output_tensor.shape().size() * output_tensor.dtype().size();
    auto                 out_data    = output_tensor.exportData<TensorDataStrided>();
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

    // Clean up
    if(use_gpu && stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    return EXIT_SUCCESS;
}
