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
#include <op_center_crop.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Center crop operation example.
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<bool>("cpu", "cpu", false, "Use CPU instead of GPU");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<int>("crop_width", "crop_width", 0, "Crop area width");
    parser.set_optional<int>("crop_height", "crop_height", 0, "Crop area height");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path  = parser.get<std::string>("i");
    std::string output_file_path = parser.get<std::string>("o");
    bool        use_gpu          = !parser.get<bool>("cpu");
    int         device_id        = parser.get<int>("d");
    int         crop_width       = parser.get<int>("crop_width");
    int         crop_height      = parser.get<int>("crop_height");
    bool        crop_set         = (crop_width > 0 && crop_height > 0);

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

    if(!crop_set)
    {
        // Set a safe crop area if no user input
        crop_width  = image_data.cols / 2;
        crop_height = image_data.rows / 2;
    }

    Size2D crop_area = {crop_width, crop_height};

    // Create input/output tensors for the image.
    TensorShape input_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                            {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType    dtype(eDataType::DATA_TYPE_U8);
    Tensor      input(input_shape, dtype, device);

    TensorShape out_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                          {1, crop_area.h, crop_area.w, image_data.channels()});
    Tensor      output(out_shape, dtype, device);

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

    CenterCrop op;
    op(stream, input, output, crop_area, device);

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
    cv::Mat output_image_data(crop_area.h, crop_area.w, image_data.type(), h_output.data());
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
    if(use_gpu)
    {
        std::cout << "Operation on GPU device " << device_id << std::endl;
    }
    else
    {
        std::cout << "Operation on CPU" << std::endl;
    }
    std::cout << "Input image size: width = " << image_data.cols << ", height = " << image_data.rows
              << std::endl;
    std::cout << "Cropping area: width = " << crop_area.w << ", height = " << crop_area.h
              << std::endl;

    // Clean up
    if(use_gpu && stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    return EXIT_SUCCESS;
}
