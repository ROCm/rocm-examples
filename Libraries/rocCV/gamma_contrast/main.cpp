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

#include <core/tensor.hpp>
#include <iostream>
#include <op_gamma_contrast.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Gamma contrast operator sample app.
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<bool>("cpu", "cpu", false, "Use CPU instead of GPU");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<float>("gamma", "gamma", 2.2f, "Gamma value to apply");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path  = parser.get<std::string>("i");
    std::string output_file_path = parser.get<std::string>("o");
    bool        use_gpu          = !parser.get<bool>("cpu");
    int         device_id        = parser.get<int>("d");
    float       gamma_value      = parser.get<float>("gamma");

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

    // Load input image using the OpenCV library.
    // The Mat image_data will store all of the data of the image
    // Image width can be gotten with image_data.cols
    // Image height can be gotten with image_data.rows
    // The amount of channels can be gotten with image_data.channels()
    cv::Mat image_data = cv::imread(input_file_path);
    if(image_data.empty())
    {
        std::cerr << "Error: Failed to read the input image file: " << input_file_path << std::endl;
        return error_exit_code;
    }

    // Batch size is needed to create the input and output tensors
    int batch_size = 1;

    // Create input/output tensors
    // Tensor shape
    //      - Takes layout as input, in this case NHWC (N - batch size, H - image height, W - image width, C - number of channels)
    //      - Also takes the datatype, in this case U8 or an unsigned integer of 8 bits.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {batch_size, image_data.rows, image_data.cols, image_data.channels()});
    DataType    dtype(eDataType::DATA_TYPE_U8);

    Tensor input(shape, dtype, device);
    Tensor output(shape, dtype, device);

    // Move image data to input tensor
    size_t image_size = image_data.rows * image_data.cols * image_data.channels() * sizeof(uint8_t);
    auto   input_data = input.exportData<TensorDataStrided>();

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(static_cast<uint8_t*>(input_data.basePtr()),
                                 image_data.data,
                                 image_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }
    else
    {
        memcpy(input_data.basePtr(), image_data.data, image_size);
    }

    // Apply gamma correction
    GammaContrast gamma_contrast;
    gamma_contrast(stream, input, output, gamma_value, device);

    // Move output data back to host
    auto                 output_data = output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(image_size);

    if(use_gpu)
    {
        HIP_CHECK(hipMemcpyAsync(h_output.data(),
                                 output_data.basePtr(),
                                 image_size,
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    else
    {
        memcpy(h_output.data(), output_data.basePtr(), image_size);
    }

    // Save the gamma-corrected image
    cv::Mat output_image(image_data.rows, image_data.cols, CV_8UC3, h_output.data());
    bool    ret = cv::imwrite(output_file_path, output_image);
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
    std::cout << "Gamma value: " << gamma_value << std::endl;

    // Clean up
    if(use_gpu && stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    return EXIT_SUCCESS;
}
