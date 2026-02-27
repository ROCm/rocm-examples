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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "rocal_api.h"
#include "rocal_api_types.h"

#define PRINT_NAMES_AND_LABELS 0

using namespace std::chrono;
std::mutex g_mtx; // mutex for critical section

/// \brief Configure command-line parser for the multi-threaded data loading example
void configure_parser(cli::Parser& parser)
{
    parser.set_optional<std::string>("i",
                                     "input",
                                     EXAMPLE_DATA_DIR,
                                     "Input image directory or path");
    parser.set_optional<bool>("g",
                              "gpu",
                              true,
                              "Use GPU processing (true) or CPU processing (false)");
    parser.set_optional<int>("ng",
                             "num_gpus",
                             1,
                             "Number of GPUs to use (only applicable when gpu=true)");
    parser.set_optional<int>("s", "num_shards", 2, "Number of data shards (threads)");
    parser.set_optional<int>("wh", "decode_width", 1024, "Decode width (0 or negative for auto)");
    parser.set_optional<int>("ht", "decode_height", 1024, "Decode height (0 or negative for auto)");
    parser.set_optional<int>("b", "batch_size", 16, "Batch size per shard");
    parser.set_optional<bool>("sh", "shuffle", false, "Shuffle the dataset");
    parser.set_optional<bool>("sv", "save_output", false, "Save output images to disk");
}

/// \brief Thread function for processing data shards
int thread_func(const char*     path,
                int             gpu_mode,
                RocalImageColor color_format,
                int             shard_id,
                int             num_shards,
                int             dec_width,
                int             dec_height,
                int             batch_size,
                bool            shuffle,
                bool            save_output)
{
    std::unique_lock<std::mutex> lck(g_mtx, std::defer_lock);
    std::cout << "Running on " << (gpu_mode >= 0 ? "GPU: " : "CPU: ") << gpu_mode << std::endl;
    std::cout << "shard_id: " << shard_id << std::endl;

    color_format              = RocalImageColor::ROCAL_COLOR_RGB24;
    int              gpu_id   = (gpu_mode < 0) ? 0 : gpu_mode;
    RocalDecoderType dec_type = RocalDecoderType::ROCAL_DECODER_TJPEG;

    lck.lock();
    // looks like OpenVX has some issue loading kernels from multiple threads at the same time
    auto handle = rocalCreate(batch_size,
                              (gpu_mode < 0) ? RocalProcessMode::ROCAL_PROCESS_CPU
                                             : RocalProcessMode::ROCAL_PROCESS_GPU,
                              gpu_id,
                              1);
    lck.unlock();

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal context"
                  << "shard_id: " << shard_id << "num_shards: " << num_shards << " >" << std::endl;
        return -1;
    }

    // Create JPEG data loader based on numshards and shard_id
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    RocalTensor decoded_output;
    if(dec_width <= 0 || dec_height <= 0)
    {
        decoded_output = rocalJpegFileSourceSingleShard(handle,
                                                        path,
                                                        color_format,
                                                        shard_id,
                                                        num_shards,
                                                        false,
                                                        shuffle,
                                                        false);
    }
    else
    {
        decoded_output = rocalJpegFileSourceSingleShard(handle,
                                                        path,
                                                        color_format,
                                                        shard_id,
                                                        num_shards,
                                                        false,
                                                        shuffle,
                                                        false,
                                                        ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                        dec_width,
                                                        dec_height,
                                                        dec_type);
    }

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "rocalJpegFileSourceSingleShard<" << shard_id << " , " << num_shards << ">"
                  << " could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Create meta data reader
    rocalCreateLabelReader(handle, path);

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalResize(handle, decoded_output, 224, 224, true);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Remaining images " << rocalGetRemainingImages(handle) << std::endl;

    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Process images and save output <<<<<<<<<<<<<<<<<*/
    int n = rocalGetAugmentationBranchCount(handle);
    int h = n * rocalGetOutputHeight(handle) * batch_size;
    int w = rocalGetOutputWidth(handle);
    int p = (((color_format == RocalImageColor::ROCAL_COLOR_RGB24)
              || (color_format == RocalImageColor::ROCAL_COLOR_RGB_PLANAR))
                 ? 3
                 : 1);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << " n "
              << n << std::endl;

    auto    cv_color_format = ((p == 3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;

    high_resolution_clock::time_point t1      = high_resolution_clock::now();
    int                               counter = 0;
    std::vector<std::string>          names;
    names.resize(batch_size);
    std::vector<int> image_name_length(batch_size);

    while(!rocalIsEmpty(handle))
    {
        if(rocalRun(handle) != 0)
        {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        // copy output to host as image
        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        unsigned          img_name_size = rocalGetImageNameLen(handle, image_name_length.data());
        std::vector<char> img_name(img_name_size);
        rocalGetImageName(handle, img_name.data());

#if PRINT_NAMES_AND_LABELS
        RocalTensorList labels = rocalGetImageLabels(handle);
        std::string     imageNamesStr(img_name.data());
        int             pos           = 0;
        int*            labels_buffer = reinterpret_cast<int*>(labels->at(0)->buffer());
        for(int i = 0; i < batch_size; i++)
        {
            names[i] = imageNamesStr.substr(pos, image_name_length[i]);
            pos += image_name_length[i];
            std::cout << "name: " << names[i] << " label: " << labels_buffer[i] << " - ";
        }
        std::cout << std::endl;
#endif

        // Save individual images if requested
        if(save_output)
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
                    std::string filename = "dataloader_shard_" + std::to_string(shard_id)
                                           + "_batch_" + std::to_string(counter / batch_size)
                                           + "_img_" + std::to_string(b) + ".png";
                    cv::imwrite(filename, mat_color);
                }
                else
                {
                    std::string filename = "dataloader_shard_" + std::to_string(shard_id)
                                           + "_batch_" + std::to_string(counter / batch_size)
                                           + "_img_" + std::to_string(b) + ".png";
                    cv::imwrite(filename, individual_image);
                }
            }
        }

        counter += batch_size;
    }

    high_resolution_clock::time_point t2           = high_resolution_clock::now();
    auto                              dur          = duration_cast<microseconds>(t2 - t1).count();
    auto                              rocal_timing = rocalGetTimingInfo(handle);

    std::cout << "For shard_id: " << shard_id << std::endl;
    std::cout << "Load     time: " << " " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time: " << " " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time: " << " " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time: " << " " << rocal_timing.transfer_time << std::endl;
    std::cout << "Processed " << counter << " images/frames." << std::endl
              << "Total Elapsed Time: " << dur / 1000000 << " sec " << dur % 1000000 << " us "
              << std::endl;

    rocalRelease(handle);
    mat_input.release();
    mat_color.release();
    return 0;
}

int main(int argc, const char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Extract parameters
    const std::string input_path       = parser.get<std::string>("i");
    const bool        use_gpu          = parser.get<bool>("g");
    const int         num_gpus         = parser.get<int>("ng");
    const int         num_shards       = parser.get<int>("s");
    const int         decode_width     = parser.get<int>("wh");
    const int         decode_height    = parser.get<int>("ht");
    const int         input_batch_size = parser.get<int>("b");
    const bool        shuffle          = parser.get<bool>("sh");
    const bool        save_output      = parser.get<bool>("sv");

    // Validate and adjust parameters
    int actual_num_gpus = num_gpus;
    if(use_gpu && num_gpus <= 0)
    {
        actual_num_gpus = 1; // Default to 1 GPU if GPU mode is enabled but count is invalid
    }
    if(!use_gpu)
    {
        actual_num_gpus = 0; // No GPUs if CPU mode is selected
    }

    // Print configuration
    std::cout << "=== rocAL Multi-threaded Data Loading Example ===" << std::endl;
    std::cout << "Processing mode: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    if(use_gpu)
    {
        std::cout << "Number of GPUs: " << actual_num_gpus << std::endl;
    }
    std::cout << "Input path: " << input_path << std::endl;
    std::cout << "Number of shards (threads): " << num_shards << std::endl;
    std::cout << "Decode dimensions: " << decode_width << "x" << decode_height;
    if(decode_width <= 0 || decode_height <= 0)
    {
        std::cout << " (auto)";
    }
    std::cout << std::endl;
    std::cout << "Batch size per shard: " << input_batch_size << std::endl;
    std::cout << "Shuffle enabled: " << (shuffle ? "yes" : "no") << std::endl;
    std::cout << "Save output: " << (save_output ? "enabled" : "disabled") << std::endl;
    std::cout << "Decoder: OpenCV" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Launch threads process shards
    std::vector<std::thread> loader_threads(num_shards);
    auto                     gpu_id = use_gpu ? 0 : -1;
    int                      th_id;

    for(th_id = 0; th_id < num_shards; th_id++)
    {
        // Distribute threads across available GPUs when GPU mode is enabled
        if(use_gpu && actual_num_gpus > 0)
        {
            gpu_id = th_id % actual_num_gpus;
        }

        loader_threads[th_id] = std::thread(thread_func,
                                            input_path.c_str(),
                                            gpu_id,
                                            RocalImageColor::ROCAL_COLOR_RGB24,
                                            th_id,
                                            num_shards,
                                            decode_width,
                                            decode_height,
                                            input_batch_size,
                                            shuffle,
                                            save_output);
    }

    for(auto& th : loader_threads)
    {
        th.join();
    }

    std::cout << "Multi-threaded data loading example completed!" << std::endl;
    return 0;
}
