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
#include <op_bnd_box.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Bounding Box operation example.
 */

/**
 * @brief Example bounding box list file content
 * 1 <-- number of images
 * 2 <-- number of boxes for image 1
 * 50 <-- X coordinate of top-left corner of box 1
 * 50 <-- Y coordinate of top-left corner of box 1
 * 100 <-- width of box 1
 * 50 <-- height of box 1
 * 5 <-- thickness of box boundary of box 1
 * 0 <-- B component of box border color of box 1
 * 0 <-- G component of box border color of box 1
 * 255 <-- R component of box border color of box 1
 * 200 <-- alpha component of box border color of box 1
 * 0 <-- B component of box fill color of box 1
 * 255 <-- G component of box fill color of box 1
 * 0 <-- R component of box fill color of box 1
 * 100 <-- alpha component of box fill color of box 1
 * 250 <-- X coordinate of top-left corner of box 2
 * 250 <-- Y coordinate of top-left corner of box 2
 * 50 <-- width of box 2
 * 100 <-- height of box 2
 * 10 <-- thickness of box boundary of box 2
 * 255 <-- B component of box border color of box 2
 * 0 <-- G component of box border color of box 2
 * 0 <-- R component of box border color of box 2
 * 200 <-- alpha component of box border color of box 2
 * 0 <-- B component of box fill color of box 2
 * 0 <-- G component of box fill color of box 2
 * 0 <-- R component of box fill color of box 2
 * 0 <-- alpha component of box fill color of box 2
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA, "Input image file path");
    parser.set_optional<std::string>("o", "output", "output.bmp", "Output image file path");
    parser.set_optional<bool>("cpu", "cpu", false, "Use CPU instead of GPU");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<std::string>("box_file", "box_file", "", "Bounding box list file");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_file_path  = parser.get<std::string>("i");
    std::string output_file_path = parser.get<std::string>("o");
    bool        use_gpu          = !parser.get<bool>("cpu");
    int         device_id        = parser.get<int>("d");
    std::string box_file_path    = parser.get<std::string>("box_file");
    bool        box_set          = !box_file_path.empty();

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

    int                                batch_size = 1;
    std::vector<std::vector<BndBox_t>> bbox_vector;

    if(box_set)
    {
        std::ifstream box_list_file(box_file_path);
        if(box_list_file.is_open())
        {
            std::string line;
            std::getline(box_list_file, line);
            batch_size = std::stoi(line.c_str());
            if(batch_size > 0)
            {
                bbox_vector.resize(batch_size);
                for(int i = 0; i < batch_size; i++)
                {
                    std::getline(box_list_file, line);
                    int num_boxes = std::stoi(line.c_str());
                    if(num_boxes > 0)
                    {
                        for(int b = 0; b < num_boxes; b++)
                        {
                            BndBox_t box;
                            std::getline(box_list_file, line);
                            box.box.x = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.y = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.width = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.height = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.thickness = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.borderColor.r = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.g = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.b = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.a = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.fillColor.r = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.g = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.b = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.a = std::atoi(line.c_str());

                            bbox_vector[i].push_back(box);
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Invalid number of boxes: " << num_boxes
                                  << " for image: " << i << std::endl;
                        return error_exit_code;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Invalid batch size: " << batch_size << std::endl;
                return error_exit_code;
            }
        }
        else
        {
            std::cerr << "Error: Failed to open bounding box list file " << box_file_path
                      << std::endl;
            return error_exit_code;
        }
    }
    else
    {
        auto width  = image_data.cols;
        auto height = image_data.rows;
        bbox_vector = {
            {
             {{width / 4, height / 4, width / 2, height / 2},
                 5,
                 {0, 0, 255, 200},
                 {0, 255, 0, 100}},
             {{width / 3, height / 3, width / 3 * 2, height / 4},
                 -1,
                 {90, 16, 181, 50},
                 {0, 0, 0, 0}},
             {{-50, (height * 2) / 3, width + 50, height / 3 + 50},
                 0,
                 {0, 0, 0, 0},
                 {111, 159, 232, 150}},
             },
        };
    }
    BndBoxes bboxes(bbox_vector);

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

    BndBox op;
    op(stream, input, output, bboxes, device);

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

    // Clean up
    if(use_gpu && stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    return EXIT_SUCCESS;
}
