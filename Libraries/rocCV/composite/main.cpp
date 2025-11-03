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
#include <op_composite.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("bg", "background", "", "Background image file path");
    parser.set_optional<std::string>("fg", "foreground", "", "Foreground image file path");
    parser.set_optional<std::string>("mask", "mask", "", "Mask image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string background_file_path = parser.get<std::string>("bg");
    std::string foreground_file_path = parser.get<std::string>("fg");
    std::string mask_file_path       = parser.get<std::string>("mask");
    std::string output_file_path     = parser.get<std::string>("o");
    int         device_id            = parser.get<int>("d");

    // Generate default images if not provided
    const int default_width  = 640;
    const int default_height = 480;

    cv::Mat background_data, foreground_data, mask_data;

    if(background_file_path.empty())
    {
        // Generate default background: blue gradient
        background_data = cv::Mat(default_height, default_width, CV_8UC3);
        for(int y = 0; y < default_height; y++)
        {
            for(int x = 0; x < default_width; x++)
            {
                background_data.at<cv::Vec3b>(y, x) = cv::Vec3b(100 + y / 3, 50 + x / 4, 200);
            }
        }
        std::cout << "Generated default background (blue gradient)" << std::endl;
    }

    if(foreground_file_path.empty())
    {
        // Generate default foreground: red circle on white background
        foreground_data = cv::Mat::ones(default_height, default_width, CV_8UC3) * 255;
        cv::Point center(default_width / 2, default_height / 2);
        cv::circle(foreground_data, center, 100, cv::Scalar(0, 0, 255), -1); // Red filled circle
        std::cout << "Generated default foreground (red circle on white)" << std::endl;
    }

    if(mask_file_path.empty())
    {
        // Generate default mask: circular gradient
        mask_data = cv::Mat::zeros(default_height, default_width, CV_8UC1);
        cv::Point center(default_width / 2, default_height / 2);
        cv::circle(mask_data, center, 100, 255, -1); // White filled circle
        cv::GaussianBlur(mask_data, mask_data, cv::Size(31, 31), 20.0); // Soft edges
        std::cout << "Generated default mask (circular gradient)" << std::endl;
    }

    // Set up device and stream
    HIP_CHECK(hipSetDevice(device_id));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Load images (only if not generated)
    if(!background_file_path.empty())
    {
        background_data = cv::imread(background_file_path);
        if(background_data.empty())
        {
            std::cerr << "Error: Failed to read background image file: " << background_file_path
                      << std::endl;
            return error_exit_code;
        }
        std::cout << "Loaded background image: " << background_file_path << std::endl;
    }

    if(!foreground_file_path.empty())
    {
        foreground_data = cv::imread(foreground_file_path);
        if(foreground_data.empty())
        {
            std::cerr << "Error: Failed to read foreground image file: " << foreground_file_path
                      << std::endl;
            return error_exit_code;
        }
        std::cout << "Loaded foreground image: " << foreground_file_path << std::endl;
    }

    if(!mask_file_path.empty())
    {
        mask_data = cv::imread(mask_file_path, cv::IMREAD_GRAYSCALE);
        if(mask_data.empty())
        {
            std::cerr << "Error: Failed to read mask image file: " << mask_file_path << std::endl;
            return error_exit_code;
        }
        std::cout << "Loaded mask image: " << mask_file_path << std::endl;
    }

    // Create input/output tensors for the image.
    DataType dtype(eDataType::DATA_TYPE_U8);

    TensorShape background_shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {1, background_data.rows, background_data.cols, background_data.channels()});
    Tensor background_tensor(background_shape, dtype);

    TensorShape foreground_shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {1, foreground_data.rows, foreground_data.cols, foreground_data.channels()});
    Tensor foreground_tensor(foreground_shape, dtype);

    TensorShape mask_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                           {1, mask_data.rows, mask_data.cols, mask_data.channels()});
    Tensor      mask_tensor(mask_shape, dtype);

    Tensor output_tensor(background_shape, dtype);

    auto bt_data = background_tensor.exportData<TensorDataStrided>();
    HIP_CHECK(hipMemcpyAsync(bt_data.basePtr(),
                             background_data.data,
                             bt_data.shape().size() * bt_data.dtype().size(),
                             hipMemcpyHostToDevice,
                             stream));

    auto ft_data = foreground_tensor.exportData<TensorDataStrided>();
    HIP_CHECK(hipMemcpyAsync(ft_data.basePtr(),
                             foreground_data.data,
                             foreground_tensor.shape().size() * foreground_tensor.dtype().size(),
                             hipMemcpyHostToDevice,
                             stream));

    auto m_data = mask_tensor.exportData<TensorDataStrided>();
    HIP_CHECK(hipMemcpyAsync(m_data.basePtr(),
                             mask_data.data,
                             mask_tensor.shape().size() * mask_tensor.dtype().size(),
                             hipMemcpyHostToDevice,
                             stream));

    hipEvent_t begin, end;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));

    HIP_CHECK(hipEventRecord(begin, stream));
    roccv::Composite op;
    op(stream, foreground_tensor, background_tensor, mask_tensor, output_tensor);
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));

    float duration;
    HIP_CHECK(hipEventElapsedTime(&duration, begin, end));
    printf("Kernel execution time: %fms\n", duration);

    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));

    // Move image data back to host
    auto                 out_data = output_tensor.exportData<TensorDataStrided>();
    std::vector<uint8_t> out_h(output_tensor.shape().size());
    HIP_CHECK(hipMemcpyAsync(out_h.data(),
                             out_data.basePtr(),
                             output_tensor.shape().size() * output_tensor.dtype().size(),
                             hipMemcpyDeviceToHost,
                             stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    // Write output image to disk
    cv::Mat output_image_data(background_data.rows, background_data.cols, CV_8UC3, out_h.data());
    bool    ret = cv::imwrite(output_file_path, output_image_data);
    if(!ret)
    {
        std::cerr << "Error: Failed to save output image to file: " << output_file_path
                  << std::endl;
        return error_exit_code;
    }

    // Print results
    std::cout << "Background: "
              << (background_file_path.empty() ? "Generated (blue gradient)" : background_file_path)
              << std::endl;
    std::cout << "Foreground: "
              << (foreground_file_path.empty() ? "Generated (red circle on white)"
                                               : foreground_file_path)
              << std::endl;
    std::cout << "Mask: "
              << (mask_file_path.empty() ? "Generated (circular gradient)" : mask_file_path)
              << std::endl;
    std::cout << "Output image file: " << output_file_path << std::endl;
    std::cout << "Operation on GPU device " << device_id << std::endl;
    std::cout << "Image size: width = " << background_data.cols
              << ", height = " << background_data.rows << std::endl;

    // Clean up
    HIP_CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}
