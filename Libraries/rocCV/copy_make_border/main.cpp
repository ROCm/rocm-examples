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
#include <iostream>
#include <op_copy_make_border.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Copy make border operation example.
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<int>("top", "top", 10, "Top border size");
    parser.set_optional<int>("left", "left", 10, "Left border size");
    parser.set_optional<float>("border_r", "border_r", 0.0f, "Border color red component");
    parser.set_optional<float>("border_g", "border_g", 0.0f, "Border color green component");
    parser.set_optional<float>("border_b", "border_b", 0.0f, "Border color blue component");
    parser.set_optional<float>("border_a", "border_a", 0.0f, "Border color alpha component");
    parser.set_optional<int>("border_mode",
                             "border_mode",
                             1,
                             "Border mode (0:constant, 1:replicate, 2:reflect, 3:wrap)");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path   = parser.get<std::string>("i");
    std::string output_file_path  = parser.get<std::string>("o");
    int         device_id         = parser.get<int>("d");
    int32_t     top               = parser.get<int>("top");
    int32_t     left              = parser.get<int>("left");
    float       border_r          = parser.get<float>("border_r");
    float       border_g          = parser.get<float>("border_g");
    float       border_b          = parser.get<float>("border_b");
    float       border_a          = parser.get<float>("border_a");
    int         border_mode_value = parser.get<int>("border_mode");

    // Validate border mode
    if(border_mode_value < 0 || border_mode_value > 3)
    {
        std::cerr << "Error: Invalid border mode. Must be 0, 1, 2, or 3." << std::endl;
        return error_exit_code;
    }
    eBorderType border_mode = static_cast<eBorderType>(border_mode_value);

    // Set up device
    HIP_CHECK(hipSetDevice(device_id));

    // Load input image
    cv::Mat image_data = cv::imread(input_file_path);
    if(image_data.empty())
    {
        std::cerr << "Error: Failed to read the input image file: " << input_file_path << std::endl;
        return error_exit_code;
    }

    // Create input/output tensors for the image.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType    dtype(eDataType::DATA_TYPE_U8);

    TensorShape o_shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {1, image_data.rows + top * 2, image_data.cols + left * 2, image_data.channels()});

    Tensor d_in(shape, dtype);
    Tensor d_out(o_shape, dtype);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Move image data to input tensor
    size_t image_size   = d_in.shape().size() * d_in.dtype().size();
    auto   d_input_data = d_in.exportData<TensorDataStrided>();
    HIP_CHECK(hipMemcpyAsync(d_input_data.basePtr(),
                             image_data.data,
                             image_size,
                             hipMemcpyHostToDevice,
                             stream));

    CopyMakeBorder op;
    op(stream, d_in, d_out, top, left, border_mode, {border_b, border_g, border_r, border_a});

    // Move image data back to host
    auto                 d_out_data     = d_out.exportData<TensorDataStrided>();
    size_t               out_image_size = d_out.shape().size() * d_out.dtype().size();
    std::vector<uint8_t> h_output(out_image_size);
    HIP_CHECK(hipMemcpyAsync(h_output.data(),
                             d_out_data.basePtr(),
                             out_image_size,
                             hipMemcpyDeviceToHost,
                             stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    // Write output image to disk
    cv::Mat output_image_data(image_data.rows + top * 2,
                              image_data.cols + left * 2,
                              CV_8UC3,
                              h_output.data());
    bool    ret = cv::imwrite(output_file_path, output_image_data);
    if(!ret)
    {
        std::cerr << "Error: Failed to save output image to file: " << output_file_path
                  << std::endl;
        return error_exit_code;
    }

    // Print results
    std::cout << "Input image file: " << input_file_path << std::endl;
    std::cout << "Output image file: " << output_file_path << std::endl;
    std::cout << "Operation on GPU device " << device_id << std::endl;
    std::cout << "Input image size: width = " << image_data.cols << ", height = " << image_data.rows
              << std::endl;
    std::cout << "Border settings: top = " << top << ", left = " << left
              << ", mode = " << border_mode_value << std::endl;

    // Clean up
    HIP_CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}
