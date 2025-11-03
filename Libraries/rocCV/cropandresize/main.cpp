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

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <op_custom_crop.hpp>
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>
#include <string>

/// \brief Writes a batch of 3-channel RGBI images in a tensor to .bmp files.
/// \param tensor A tensor containing a batch of RGBI images.
/// \param stream The HIP stream to synchronize with.
inline void write_rgb_i_tensor(const roccv::Tensor& tensor, hipStream_t stream)
{
    HIP_CHECK(hipStreamSynchronize(stream));

    auto src_data   = tensor.exportData<roccv::TensorDataStrided>();
    int  batch_size = tensor.shape(tensor.layout().batch_index());
    int  height     = tensor.shape(tensor.layout().height_index());
    int  width      = tensor.shape(tensor.layout().width_index());

    // Write each image in the batch to separate .bmp files
    for(int b = 0; b < batch_size; b++)
    {
        std::ostringstream out_filename;
        out_filename << "./roccvtest_" << b << ".bmp";

        cv::Mat rgb_output_mat(height, width, CV_8UC3);
        HIP_CHECK(hipMemcpy(rgb_output_mat.data,
                            src_data.basePtr(),
                            (tensor.shape().size() / batch_size) * tensor.dtype().size(),
                            hipMemcpyDeviceToHost));

        // Convert RGB back to BGR for BMP file format
        cv::Mat bgr_output_mat;
        cv::cvtColor(rgb_output_mat, bgr_output_mat, cv::COLOR_RGB2BGR);
        cv::imwrite(out_filename.str().c_str(), bgr_output_mat);
    }
}

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * rocCV Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);

    // Set up command line arguments
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA_DIR, "Image directory path");
    parser.set_optional<int>("b", "batch", 1, "Batch size");

    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string image_path = parser.get<std::string>("i");
    uint32_t    batch_size = parser.get<int>("b");

    // Validate arguments
    if(image_path.empty())
    {
        std::cerr << "Error: Image path is required." << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " --input <image_path_or_directory> [--batch <batch_size>]" << std::endl;
        return error_exit_code;
    }

    std::ifstream image_file(image_path);
    if(!image_file.good())
    {
        std::cerr << "Error: Image path '" + image_path + "' does not exist" << std::endl;
        return error_exit_code;
    }

    // First, load the image to determine its dimensions dynamically
    std::cout << "Loading image to determine dimensions..." << std::endl;

    // Handle both directory and single file paths
    std::vector<std::string> image_file_paths;
    if(std::filesystem::is_directory(image_path))
    {
        // Collect all JPEG files in the directory
        for(auto file : std::filesystem::directory_iterator(image_path))
        {
            if(!std::filesystem::is_directory(file.path())
               && (file.path().extension() == ".jpg" || file.path().extension() == ".jpeg"))
            {
                image_file_paths.push_back(file.path());
            }
        }
        if(image_file_paths.empty())
        {
            std::cerr << "Error: No JPEG files found in directory " << image_path << std::endl;
            return error_exit_code;
        }
        std::cout << "Found " << image_file_paths.size() << " JPEG files in directory" << std::endl;

        // Validate that all images have the same dimensions
        std::cout << "Validating image dimensions..." << std::endl;
        cv::Mat first_image = cv::imread(image_file_paths[0]);
        if(first_image.empty())
        {
            std::cerr << "Error: Unable to load first image " << image_file_paths[0] << std::endl;
            return error_exit_code;
        }
        int expected_width  = first_image.cols;
        int expected_height = first_image.rows;

        for(const auto& file : image_file_paths)
        {
            cv::Mat test_image = cv::imread(file);
            if(test_image.empty())
            {
                std::cerr << "Error: Unable to load image " << file << std::endl;
                return error_exit_code;
            }
            if(test_image.cols != expected_width || test_image.rows != expected_height)
            {
                std::cerr << "Error: Image " << file << " has dimensions " << test_image.cols << "x"
                          << test_image.rows << " but expected " << expected_width << "x"
                          << expected_height
                          << " (all images in directory must have the same dimensions)"
                          << std::endl;
                return error_exit_code;
            }
        }
        std::cout << "All images have consistent dimensions: " << expected_width << "x"
                  << expected_height << std::endl;
    }
    else
    {
        image_file_paths.push_back(image_path);
    }

    // Validate batch size against number of images
    if(batch_size > image_file_paths.size())
    {
        std::cout << "Warning: Batch size (" << batch_size << ") exceeds number of images ("
                  << image_file_paths.size() << "), reducing batch size to "
                  << image_file_paths.size() << std::endl;
        batch_size = image_file_paths.size();
    }

    // Load first image to determine dimensions
    cv::Mat input_mat = cv::imread(image_file_paths[0]);
    if(input_mat.empty())
    {
        std::cerr << "Error: Unable to load image " << image_file_paths[0] << std::endl;
        return error_exit_code;
    }

    std::cout << "Loaded image: " << input_mat.cols << "x" << input_mat.rows << " with "
              << input_mat.channels() << " channels" << std::endl;

    // Convert BGR to RGB if needed (OpenCV loads JPEGs as BGR by default)
    cv::Mat rgb_mat;
    if(input_mat.channels() == 3)
    {
        cv::cvtColor(input_mat, rgb_mat, cv::COLOR_BGR2RGB);
        std::cout << "Converted BGR to RGB format" << std::endl;
    }
    else
    {
        rgb_mat = input_mat.clone();
    }

    // Ensure we have 3 channels for RGB format
    if(rgb_mat.channels() != 3)
    {
        std::cerr << "Error: Image must have 3 channels (RGB), but has " << rgb_mat.channels()
                  << " channels" << std::endl;
        return error_exit_code;
    }

    // Set dimensions dynamically based on the loaded image
    int max_image_width  = rgb_mat.cols;
    int max_image_height = rgb_mat.rows;
    int max_channels     = 3;

    std::cout << "Using dynamic dimensions: " << max_image_width << "x" << max_image_height
              << std::endl;

    // Create the HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate input tensor
    // Allocating memory for RGBI input image batch of uint8_t data type.
    roccv::TensorDataStrided::Buffer in_buf;
    in_buf.strides[3] = sizeof(uint8_t);
    in_buf.strides[2] = max_channels * in_buf.strides[3];
    in_buf.strides[1] = max_image_width * in_buf.strides[2];
    in_buf.strides[0] = max_image_height * in_buf.strides[1];
    HIP_CHECK(hipMallocAsync(&in_buf.basePtr, batch_size * in_buf.strides[0], stream));

    // Tensor Requirements
    // Calculate the requirements for the RGBI uint8_t Tensor which include
    // pitch bytes, alignment, shape  and tensor layout
    roccv::Tensor::Requirements in_reqs
        = roccv::Tensor::CalcRequirements(batch_size,
                                          {max_image_width, max_image_height},
                                          roccv::FMT_RGB8);

    // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    roccv::TensorDataStrided in_data(in_reqs.shape, in_reqs.dtype, in_buf);

    // Wrap tensor data in a rocCV tensor for use with the rocCV operators.
    roccv::Tensor in_tensor = roccv::TensorWrapData(in_data);

    // Image Loading with explicit OpenCV handling for JPEG files
    uint8_t* gpu_input = reinterpret_cast<uint8_t*>(in_buf.basePtr);

    // Load and copy all images for batch processing
    size_t image_size = rgb_mat.rows * rgb_mat.cols * rgb_mat.channels() * sizeof(uint8_t);
    size_t mem_offset = 0;

    std::cout << "Loading " << batch_size << " images for batch processing..." << std::endl;

    for(uint32_t i = 0; i < batch_size; i++)
    {
        cv::Mat batch_image = cv::imread(image_file_paths[i]);
        if(batch_image.empty())
        {
            std::cerr << "Error: Unable to load image " << image_file_paths[i] << std::endl;
            return error_exit_code;
        }

        // Convert BGR to RGB if needed
        cv::Mat batch_rgb_mat;
        if(batch_image.channels() == 3)
        {
            cv::cvtColor(batch_image, batch_rgb_mat, cv::COLOR_BGR2RGB);
        }
        else
        {
            batch_rgb_mat = batch_image.clone();
        }

        // Ensure we have 3 channels for RGB format
        if(batch_rgb_mat.channels() != 3)
        {
            std::cerr << "Error: Image " << image_file_paths[i]
                      << " must have 3 channels (RGB), but has " << batch_rgb_mat.channels()
                      << " channels" << std::endl;
            return error_exit_code;
        }

        // Copy RGB image data to GPU memory at correct offset
        HIP_CHECK(hipMemcpyAsync(gpu_input + mem_offset,
                                 batch_rgb_mat.data,
                                 image_size,
                                 hipMemcpyHostToDevice,
                                 stream));
        mem_offset += image_size;
    }

    // Wait for all copies to complete before proceeding
    HIP_CHECK(hipStreamSynchronize(stream));

    std::cout << "Successfully loaded and copied " << batch_size << " images to GPU memory"
              << std::endl;

    // Set parameters for Center Crop and Resize
    // Calculate the largest square that can be cropped from the center
    int crop_size   = std::min(max_image_width, max_image_height);
    int crop_x      = (max_image_width - crop_size) / 2;
    int crop_y      = (max_image_height - crop_size) / 2;
    int crop_width  = crop_size;
    int crop_height = crop_size;

    std::cout << "Center cropping: (" << crop_x << "," << crop_y << ") with size " << crop_width
              << "x" << crop_height << std::endl;

    // Set the resize dimensions (square to match square crop)
    int resize_width  = 320;
    int resize_height = 320;

    // Create the crop rect for the cropping operator
    roccv::Box_t crop_rect = {crop_x, crop_y, crop_width, crop_height};

    // Allocate Tensors for Crop and Resize
    // Create a rocCV Tensor based on the crop window size.
    roccv::Tensor crop_tensor(batch_size, {crop_width, crop_height}, roccv::FMT_RGB8);
    // Create a rocCV Tensor based on resize dimensions
    roccv::Tensor resized_tensor(batch_size, {resize_width, resize_height}, roccv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
#endif

    // Initialize operators for Crop and Resize
    roccv::CustomCrop crop_op;
    roccv::Resize     resize_op;

    // Executes the CustomCrop operation on the given HIP stream
    crop_op(stream, in_tensor, crop_tensor, crop_rect);

    // Resize operator can now be enqueued into the same stream
    resize_op(stream, crop_tensor, resized_tensor, INTERP_TYPE_LINEAR);

    // Profile section
#ifdef PROFILE_SAMPLE
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float operatorms = 0;
    hipEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // Copy the buffer to CPU and write resized image into .bmp files
    write_rgb_i_tensor(resized_tensor, stream);

    // Clean up
    HIP_CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}
