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

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "rocal_api.h"

/// \brief Configure command-line parser for the basic rocAL example
void configure_parser(cli::Parser& parser)
{
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA_DIR, "Input image directory path");
    parser.set_optional<int>("b", "batch_size", 4, "Batch size for processing");
    parser.set_optional<int>("wh", "width", 224, "Output image width");
    parser.set_optional<int>("ht", "height", 224, "Output image height");
    parser.set_optional<bool>("g",
                              "gpu",
                              true,
                              "Use GPU processing (true) or CPU processing (false)");
    parser.set_optional<bool>("c", "rgb", true, "Process RGB images (true) or grayscale (false)");
    parser.set_optional<int>("s", "shards", 1, "Number of decode shards");
    parser.set_optional<bool>("d",
                              "dynamic_mode",
                              true,
                              "Use dynamic mode (process all images) vs fixed mode");
}

/// \brief Set up the basic rocAL augmentation pipeline
bool setup_augmentation_pipeline(RocalContext       handle,
                                 const std::string& input_path,
                                 int                output_width,
                                 int                output_height,
                                 bool               use_rgb,
                                 int                num_shards)
{
    // Determine color format based on user preference
    RocalImageColor color_format
        = (use_rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    // Set the rocAL decoder type (OpenCV only for simplicity)
    RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG;

    std::cout << "Loading images from: " << input_path << std::endl;

    // Create JPEG file source with optional size constraints
    // The loader can automatically select the best size to decode all images
    RocalTensor decoded_output;
    if(output_height <= 0 || output_width <= 0)
    {
        decoded_output = rocalJpegFileSource(handle,
                                             input_path.c_str(),
                                             color_format,
                                             num_shards,
                                             false,
                                             false);
    }
    else
    {
        decoded_output = rocalJpegFileSource(handle,
                                             input_path.c_str(),
                                             color_format,
                                             num_shards,
                                             false,
                                             false,
                                             false,
                                             ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                             output_width,
                                             output_height,
                                             rocal_decoder_type);
    }

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle)
                  << std::endl;
        return false;
    }

    // Apply crop and resize augmentation
    rocalCropResizeFixed(handle, decoded_output, 224, 224, true, 0.9, 1.1, 0.1, 0.1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
        return false;
    }

    return true;
}

/// \brief Process images and save results individually
void process_augmented_images(RocalContext handle, int batch_size, bool use_dynamic_mode)
{
    // Get output dimensions
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * batch_size;
    int w = rocalGetOutputWidth(handle);
    int p = ((rocalGetOutputColorFormat(handle) == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);

    std::cout << "output width " << w << " output height " << h << " color planes " << p
              << std::endl;
    auto cv_color_format = ((p == 3) ? CV_8UC3 : CV_8UC1);

    const int        total_batches = 2;
    int              batch_id      = -1;
    std::vector<int> image_name_lengths(batch_size);
    // Fixed run lengths for non-dynamic mode
    int batch_lengths[] = {2 * batch_size, 4 * batch_size};

    std::vector<std::string> names;
    names.resize(batch_size);

    while(++batch_id < total_batches)
    {
        std::cout << "Start batch id " << batch_id << "\n";
        std::cout << "Available images = " << rocalGetRemainingImages(handle) << std::endl;

        // Determine how many images to process based on mode
        int process_image_count;
        if(use_dynamic_mode)
        {
            process_image_count = rocalGetRemainingImages(handle);
        }
        else
        {
            process_image_count = batch_lengths[batch_id];
        }
        std::cout << "Process " << process_image_count << " images" << std::endl;

        cv::Mat mat_input(h, w, cv_color_format);
        cv::Mat mat_color;

        int counter = 0;

        // Loop condition depends on mode
        while((use_dynamic_mode ? !rocalIsEmpty(handle) : (counter < batch_lengths[batch_id])))
        {
            if(rocalRun(handle) != 0)
            {
                std::cout << "rocalRun Failed with runtime error" << std::endl;
                rocalRelease(handle);
                return;
            }

            rocalCopyToOutput(handle, mat_input.data, h * w * p);

            counter += batch_size;

            unsigned imagename_size = rocalGetImageNameLen(handle, image_name_lengths.data());
            std::vector<char> imageNames(imagename_size);
            rocalGetImageName(handle, imageNames.data());
            std::string imageNamesStr(imageNames.data());

            int pos = 0;
            for(int i = 0; i < batch_size; i++)
            {
                names[i] = imageNamesStr.substr(pos, image_name_lengths[i]);
                pos += image_name_lengths[i];
                std::cout << "name: " << names[i] << std::endl;
            }
            std::cout << std::endl;

            // Save individual images
            {
                // Calculate dimensions for individual images
                int single_h = h / batch_size;
                int single_w = w;

                // Save each image in the batch separately
                for(int b = 0; b < batch_size; b++)
                {
                    // Extract individual image from the batch
                    cv::Rect roi(0, b * single_h, single_w, single_h);
                    cv::Mat  individual_image = mat_input(roi);

                    if(rocalGetOutputColorFormat(handle) == RocalImageColor::ROCAL_COLOR_RGB24)
                    {
                        cv::cvtColor(individual_image, mat_color, cv::COLOR_RGB2BGR);
                        std::string filename = "basic_batch_id_" + std::to_string(batch_id)
                                               + "_iter_" + std::to_string(counter / batch_size)
                                               + "_img_" + std::to_string(b) + ".png";
                        cv::imwrite(filename, mat_color);
                    }
                    else
                    {
                        std::string filename = "basic_batch_id_" + std::to_string(batch_id)
                                               + "_iter_" + std::to_string(counter / batch_size)
                                               + "_img_" + std::to_string(b) + ".png";
                        cv::imwrite(filename, individual_image);
                    }
                }
            }
        }
        std::cout << "Completed batch id: " << batch_id << " processed " << counter << " images\n";
        std::cout << "rocAL reset\n";
        rocalResetLoaders(handle);
        mat_input.release();
        mat_color.release();
    }
}

int main(int argc, const char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Extract parameters
    const std::string input_path    = parser.get<std::string>("i");
    const int         batch_size    = parser.get<int>("b");
    const int         output_width  = parser.get<int>("wh");
    const int         output_height = parser.get<int>("ht");
    const bool        use_gpu       = parser.get<bool>("g");
    const bool        use_rgb       = parser.get<bool>("c");
    const int         num_shards    = parser.get<int>("s");
    const bool        dynamic_mode  = parser.get<bool>("d");

    // Print configuration
    std::cout << ">>> Running on " << (use_gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Output dimensions: " << output_width << "x" << output_height << std::endl;
    std::cout << "Color format: " << (use_rgb ? "RGB" : "Grayscale") << std::endl;
    std::cout << "Number of shards: " << num_shards << std::endl;

    // Create rocAL context
    RocalProcessMode process_mode
        = use_gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU;

    auto handle = rocalCreate(batch_size, process_mode, 0, 1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    // Set up the augmentation pipeline
    if(!setup_augmentation_pipeline(handle,
                                    input_path,
                                    output_width,
                                    output_height,
                                    use_rgb,
                                    num_shards))
    {
        rocalRelease(handle);
        return -1;
    }

    // Verify and build the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        rocalRelease(handle);
        return -1;
    }

    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    // Process images
    process_augmented_images(handle, batch_size, dynamic_mode);

    // Clean up
    rocalRelease(handle);

    std::cout << "Basic processing example completed successfully!" << std::endl;
    return 0;
}
