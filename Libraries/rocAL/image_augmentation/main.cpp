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

using namespace std::chrono;

/// \brief Configure command-line parser for the image augmentation example
void configure_parser(cli::Parser& parser)
{
    parser.set_optional<std::string>("i", "input", EXAMPLE_DATA_DIR, "Input image directory path");
    parser.set_optional<bool>("g",
                              "gpu",
                              true,
                              "Use GPU processing (true) or CPU processing (false)");
    parser.set_optional<bool>("c", "rgb", true, "Process RGB images (true) or grayscale (false)");
    parser.set_optional<int>("wh", "width", 0, "Output image width (0 = auto)");
    parser.set_optional<int>("ht", "height", 0, "Output image height (0 = auto)");
    parser.set_optional<int>("b", "batch_size", 4, "Batch size for processing");
    parser.set_optional<int>("s", "shards", 2, "Number of decode shards");
    parser.set_optional<bool>("sh", "shuffle", false, "Shuffle the dataset");
    parser.set_optional<int>("ad", "aug_depth", 1, "Augmentation depth (number of blur passes)");
    parser.set_optional<bool>("sv", "save_output", false, "Save augmented images to disk");
}

/// \brief Set up the image augmentation pipeline
bool setup_augmentation_pipeline(RocalContext       handle,
                                 const std::string& input_path,
                                 int                output_width,
                                 int                output_height,
                                 bool               use_rgb,
                                 int                num_shards,
                                 bool               shuffle,
                                 int                aug_depth)
{
    RocalImageColor color_format
        = use_rgb ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    std::cout << "Loading images from: " << input_path << std::endl;
    std::cout << "Augmentation depth: " << aug_depth << std::endl;
    std::cout << "Decoder: OpenCV" << std::endl;

    /*>>>>>>>>>>>>>>>> Creating rocAL parameters  <<<<<<<<<<<<<<<<*/

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalFloatParam rand_crop_area = rocalCreateFloatUniformRand(0.3, 0.5);
    auto            status         = rocalUpdateFloatUniformRand(0.2, 0.5, rand_crop_area);
    RocalIntParam   color_temp_adj = rocalCreateIntParameter(0);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values              = 3;
    float        values[num_values]      = {0, 10, 135};
    double       frequencies[num_values] = {1, 5, 5};

    RocalFloatParam rand_angle = rocalCreateFloatRand(values, frequencies, num_values);
    status                     = rocalUpdateFloatRand(values, frequencies, num_values, rand_angle);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

    // Create JPEG file source with OpenCV decoder only
    RocalTensor input_tensor;
    if(output_height <= 0 || output_width <= 0)
    {
        input_tensor = rocalJpegFileSource(handle,
                                           input_path.c_str(),
                                           color_format,
                                           num_shards,
                                           false,
                                           shuffle,
                                           false);
    }
    else
    {
        input_tensor = rocalJpegFileSource(handle,
                                           input_path.c_str(),
                                           color_format,
                                           num_shards,
                                           false,
                                           shuffle,
                                           false,
                                           ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                           output_width,
                                           output_height,
                                           RocalDecoderType::ROCAL_DECODER_TJPEG);
    }

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "JPEG source could not initialize: " << rocalGetErrorMessage(handle)
                  << std::endl;
        return false;
    }

    RocalTensor tensor0;
    int         resize_w = 112, resize_h = 112;
    tensor0 = rocalResize(handle, input_tensor, resize_w, resize_h, true);

    // Main augmentation branch: rain -> fisheye -> rotate
    RocalTensor tensor1  = rocalRain(handle, tensor0, false);
    RocalTensor tensor11 = rocalFishEye(handle, tensor1, false);
    rocalRotate(handle, tensor11, true, rand_angle);

    // Creating successive blur nodes to simulate a deep branch of augmentations
    RocalTensor tensor2
        = rocalCropResize(handle, tensor0, resize_w, resize_h, false, rand_crop_area);
    for(int i = 0; i < aug_depth; i++)
    {
        tensor2 = rocalBlur(handle, tensor2, (i == (aug_depth - 1)) ? true : false);
    }

    // Additional augmentations: snow -> blend -> exposure
    RocalTensor tensor8 = rocalSnow(handle, tensor0, true);
    RocalTensor tensor9 = rocalBlend(handle, tensor0, tensor8, true);
    rocalExposure(handle, tensor9, true);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Failed to build augmentation pipeline: " << rocalGetErrorMessage(handle)
                  << std::endl;
        return false;
    }

    return true;
}

/// \brief Process images with augmentation and save results if requested
void process_augmented_images(RocalContext handle, bool save_output, int batch_size)
{
    // Get output dimensions
    int output_height
        = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * batch_size;
    int output_width = rocalGetOutputWidth(handle);
    int color_planes
        = (rocalGetOutputColorFormat(handle) == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1;

    std::cout << "Output width: " << output_width << ", output height: " << output_height
              << ", color planes: " << color_planes << std::endl;

    auto    cv_color_format = (color_planes == 3) ? CV_8UC3 : CV_8UC1;
    cv::Mat mat_input(output_height, output_width, cv_color_format);
    cv::Mat mat_color;

    high_resolution_clock::time_point t1                   = high_resolution_clock::now();
    int                               counter              = 0;
    int                               color_temp_increment = 1;

    // Create color temperature adjustment parameter for dynamic augmentation
    RocalIntParam color_temp_adj = rocalCreateIntParameter(0);

    while(!rocalIsEmpty(handle))
    {
        if(rocalRun(handle) != 0)
        {
            std::cerr << "rocalRun failed with runtime error" << std::endl;
            return;
        }

        // Update color temperature parameter dynamically
        if(rocalGetIntValue(color_temp_adj) <= -99 || rocalGetIntValue(color_temp_adj) >= 99)
        {
            color_temp_increment *= -1;
        }
        rocalUpdateIntParameter(rocalGetIntValue(color_temp_adj) + color_temp_increment,
                                color_temp_adj);

        // Get output tensors and copy to host buffer
        auto output_tensor_list = rocalGetOutputTensors(handle);
        if(!output_tensor_list || output_tensor_list->size() == 0)
        {
            std::cerr << "Warning: No output tensors available" << std::endl;
            continue;
        }

        unsigned char* output = mat_input.data;
        for(unsigned int i = 0; i < output_tensor_list->size(); i++)
        {
            output_tensor_list->at(i)->copy_data(output);
            output += output_tensor_list->at(i)->data_size();
        }

        // Save augmented images individually if requested
        if(save_output)
        {
            int single_height = rocalGetOutputHeight(handle);
            int single_width  = output_width;
            int num_branches  = rocalGetAugmentationBranchCount(handle);
            int total_images  = num_branches * batch_size;
            int batch_number  = counter / batch_size;

            // Save each augmented image separately
            for(int i = 0; i < total_images; i++)
            {
                cv::Rect roi(0, i * single_height, single_width, single_height);
                cv::Mat  individual_image = mat_input(roi);

                if(rocalGetOutputColorFormat(handle) == RocalImageColor::ROCAL_COLOR_RGB24)
                {
                    cv::cvtColor(individual_image, mat_color, cv::COLOR_RGB2BGR);

                    int branch    = i / batch_size;
                    int batch_idx = i % batch_size;

                    std::string filename = "aug_batch_" + std::to_string(batch_number) + "_branch_"
                                           + std::to_string(branch) + "_img_"
                                           + std::to_string(batch_idx) + ".png";
                    cv::imwrite(filename, mat_color);
                }
                else
                {
                    int branch    = i / batch_size;
                    int batch_idx = i % batch_size;

                    std::string filename = "aug_batch_" + std::to_string(batch_number) + "_branch_"
                                           + std::to_string(branch) + "_img_"
                                           + std::to_string(batch_idx) + ".png";
                    cv::imwrite(filename, individual_image);
                }
            }
        }

        counter += batch_size;
    }

    high_resolution_clock::time_point t2       = high_resolution_clock::now();
    auto                              duration = duration_cast<microseconds>(t2 - t1).count();

    std::cout << "\nProcessed " << counter << " images." << std::endl;
    std::cout << "Total elapsed time: " << duration / 1000000 << " sec " << duration % 1000000
              << " us" << std::endl;

    // Print timing information
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "\nDetailed timing:" << std::endl;
    std::cout << "  Load time:     " << rocal_timing.load_time << " ms" << std::endl;
    std::cout << "  Decode time:   " << rocal_timing.decode_time << " ms" << std::endl;
    std::cout << "  Process time:  " << rocal_timing.process_time << " ms" << std::endl;
    std::cout << "  Transfer time: " << rocal_timing.transfer_time << " ms" << std::endl;

    if(save_output)
    {
        std::cout << "\nAugmented images saved to current directory" << std::endl;
    }

    mat_input.release();
    mat_color.release();
}

int main(int argc, const char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Extract parameters
    const std::string input_path    = parser.get<std::string>("i");
    const bool        use_gpu       = parser.get<bool>("g");
    const bool        use_rgb       = parser.get<bool>("c");
    const int         output_width  = parser.get<int>("wh");
    const int         output_height = parser.get<int>("ht");
    const int         batch_size    = parser.get<int>("b");
    const int         num_shards    = parser.get<int>("s");
    const bool        shuffle       = parser.get<bool>("sh");
    const int         aug_depth     = parser.get<int>("ad");
    const bool        save_output   = parser.get<bool>("sv");

    // Print configuration
    std::cout << "=== rocAL Image Augmentation Example ===" << std::endl;
    std::cout << "Running on: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "Input path: " << input_path << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Output dimensions: " << output_width << "x" << output_height
              << (output_width == 0 ? " (auto)" : "") << std::endl;
    std::cout << "Color format: " << (use_rgb ? "RGB" : "Grayscale") << std::endl;
    std::cout << "Number of shards: " << num_shards << std::endl;
    std::cout << "Shuffle enabled: " << (shuffle ? "yes" : "no") << std::endl;
    std::cout << "Augmentation depth: " << aug_depth << std::endl;
    std::cout << "Decoder: OpenCV" << std::endl;
    std::cout << "Save output: " << (save_output ? "enabled" : "disabled") << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create rocAL context
    RocalProcessMode process_mode
        = use_gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU;
    auto handle = rocalCreate(batch_size, process_mode, 0, 1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Could not create the rocAL context" << std::endl;
        return -1;
    }

    // Set up the augmentation pipeline
    if(!setup_augmentation_pipeline(handle,
                                    input_path,
                                    output_width,
                                    output_height,
                                    use_rgb,
                                    num_shards,
                                    shuffle,
                                    aug_depth))
    {
        rocalRelease(handle);
        return -1;
    }

    // Verify the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cerr << "Could not verify the augmentation graph" << std::endl;
        rocalRelease(handle);
        return -1;
    }

    std::cout << "Remaining images: " << rocalGetRemainingImages(handle) << std::endl;
    std::cout << "Augmentation branches: " << rocalGetAugmentationBranchCount(handle) << std::endl;

    // Process images with augmentation
    process_augmented_images(handle, save_output, batch_size);

    // Clean up
    rocalRelease(handle);

    std::cout << "\nImage augmentation example completed successfully!" << std::endl;
    return 0;
}
