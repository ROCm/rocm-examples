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
#include "rocjpeg_utils.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>(
        "input",
        "input",
        EXAMPLE_DATA_DIR,
        "Input path to a single JPEG image or a directory containing JPEG images");
    parser.set_optional<std::string>("output",
                                     "output",
                                     "",
                                     "Output path to a file or directory to save decoded images");
    parser.set_optional<int>("device", "device", 0, "Device ID");
    parser.set_optional<int>(
        "backend",
        "backend",
        0,
        "rocJPEG backend (0 for hardware-accelerated, 1 for hybrid - currently not supported)");
    parser.set_optional<std::string>("format",
                                     "format",
                                     "native",
                                     "Output format: native, yuv_planar, y, rgb, rgb_planar");
    parser.set_optional<int>("batch_size", "batch_size", 2, "Batch size for decoding");
    parser.set_optional<std::string>("crop", "crop", "", "Crop rectangle: left,top,right,bottom");
    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_path       = parser.get<std::string>("input");
    std::string output_file_path = parser.get<std::string>("output");
    bool        save_images      = !output_file_path.empty();
    int         device_id        = parser.get<int>("device");
    int         backend_int      = parser.get<int>("backend");
    std::string format_str       = parser.get<std::string>("format");
    int         batch_size       = parser.get<int>("batch_size");
    std::string crop_str         = parser.get<std::string>("crop");

    // Convert backend to enum
    RocJpegBackend rocjpeg_backend = static_cast<RocJpegBackend>(backend_int);

    // Set up decode parameters
    RocJpegDecodeParams decode_params = {};
    if(format_str == "native")
    {
        decode_params.output_format = ROCJPEG_OUTPUT_NATIVE;
    }
    else if(format_str == "yuv_planar")
    {
        decode_params.output_format = ROCJPEG_OUTPUT_YUV_PLANAR;
    }
    else if(format_str == "y")
    {
        decode_params.output_format = ROCJPEG_OUTPUT_Y;
    }
    else if(format_str == "rgb")
    {
        decode_params.output_format = ROCJPEG_OUTPUT_RGB;
    }
    else if(format_str == "rgb_planar")
    {
        decode_params.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
    }
    else
    {
        std::cerr << "ERROR: Invalid output format: " << format_str << std::endl;
        return EXIT_FAILURE;
    }

    // Parse crop rectangle if provided
    if(!crop_str.empty())
    {
        if(4
           != sscanf(crop_str.c_str(),
                     "%hd,%hd,%hd,%hd",
                     &decode_params.crop_rectangle.left,
                     &decode_params.crop_rectangle.top,
                     &decode_params.crop_rectangle.right,
                     &decode_params.crop_rectangle.bottom))
        {
            std::cerr << "ERROR: Invalid crop rectangle format" << std::endl;
            return EXIT_FAILURE;
        }
        int crop_width  = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
        int crop_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;
        if(crop_width % 2 == 1 || crop_height % 2 == 1)
        {
            std::cerr << "ERROR: Crop rectangle must have width and height of even numbers"
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Get file paths
    std::vector<std::string> file_paths;
    bool                     is_dir  = false;
    bool                     is_file = false;
    if(!rocjpeg_utils::get_file_paths(input_path, file_paths, is_dir, is_file))
    {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize HIP device
    if(!rocjpeg_utils::init_hip_device(device_id))
    {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create rocJPEG handle
    RocJpegHandle rocjpeg_handle = nullptr;
    ROCJPEG_CHECK(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));

    // Adjust batch size
    batch_size = std::min(batch_size, static_cast<int>(file_paths.size()));

    // Create stream handles
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles(batch_size);
    for(int i = 0; i < batch_size; i++)
    {
        ROCJPEG_CHECK(rocJpegStreamCreate(&rocjpeg_stream_handles[i]));
    }

    // Initialize batch processing variables
    std::vector<std::vector<char>>     batch_images(batch_size);
    std::vector<RocJpegImage>          output_images(batch_size);
    std::vector<RocJpegDecodeParams>   decode_params_batch(batch_size, decode_params);
    std::vector<std::vector<uint32_t>> prior_channel_sizes(
        batch_size,
        std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>>    widths(batch_size,
                                              std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>>    heights(batch_size,
                                               std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<RocJpegChromaSubsampling> subsamplings(batch_size);
    std::vector<std::string>              base_file_names(batch_size);
    std::vector<RocJpegStreamHandle>      rocjpeg_stream_handles_for_current_batch(batch_size);

    // Statistics
    int      total_images                          = 0;
    uint64_t num_bad_jpegs                         = 0;
    uint64_t num_jpegs_with_411_subsampling        = 0;
    uint64_t num_jpegs_with_unknown_subsampling    = 0;
    uint64_t num_jpegs_with_unsupported_resolution = 0;
    double   time_per_image_all                    = 0;
    double   mpixels_all                           = 0;

    rocjpeg_utils utils;
    std::cout << "Decoding started, please wait! ... " << std::endl;

    // Process images in batches
    for(int i = 0; i < static_cast<int>(file_paths.size()); i += batch_size)
    {
        int batch_end          = std::min(i + batch_size, static_cast<int>(file_paths.size()));
        int current_batch_size = 0;

        for(int j = i; j < batch_end; j++)
        {
            int         index = j - i;
            std::string temp_base_file_name
                = file_paths[j].substr(file_paths[j].find_last_of("/\\") + 1);

            // Read image from disk
            std::ifstream input(file_paths[j].c_str(),
                                std::ios::in | std::ios::binary | std::ios::ate);
            if(!input.is_open())
            {
                std::cerr << "ERROR: Cannot open image: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }

            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);

            if(batch_images[index].size() < static_cast<size_t>(file_size))
            {
                batch_images[index].resize(file_size);
            }

            if(!input.read(batch_images[index].data(), file_size))
            {
                std::cerr << "ERROR: Cannot read from file: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }

            // Parse JPEG stream
            RocJpegStatus rocjpeg_status
                = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_images[index].data()),
                                     file_size,
                                     rocjpeg_stream_handles[index]);
            if(rocjpeg_status != ROCJPEG_STATUS_SUCCESS)
            {
                if(is_dir)
                {
                    num_bad_jpegs++;
                    std::cerr << "Skipping decoding input file: " << file_paths[j] << std::endl;
                    continue;
                }
                else
                {
                    std::cerr << "ERROR: Failed to parse the input jpeg stream with "
                              << rocJpegGetErrorName(rocjpeg_status) << std::endl;
                    return EXIT_FAILURE;
                }
            }

            // Get image info
            uint8_t                  num_components;
            std::vector<uint32_t>    temp_widths(ROCJPEG_MAX_COMPONENT, 0);
            std::vector<uint32_t>    temp_heights(ROCJPEG_MAX_COMPONENT, 0);
            RocJpegChromaSubsampling temp_subsampling;

            ROCJPEG_CHECK(rocJpegGetImageInfo(rocjpeg_handle,
                                              rocjpeg_stream_handles[index],
                                              &num_components,
                                              &temp_subsampling,
                                              temp_widths.data(),
                                              temp_heights.data()));

            // Check resolution
            if(temp_widths[0] < 64 || temp_heights[0] < 64)
            {
                if(is_dir)
                {
                    num_jpegs_with_unsupported_resolution++;
                    continue;
                }
                else
                {
                    std::cerr << "ERROR: The image resolution is not supported by VCN Hardware"
                              << std::endl;
                    return EXIT_FAILURE;
                }
            }

            // Check subsampling
            if(temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN)
            {
                if(is_dir)
                {
                    if(temp_subsampling == ROCJPEG_CSS_411)
                    {
                        num_jpegs_with_411_subsampling++;
                    }
                    if(temp_subsampling == ROCJPEG_CSS_UNKNOWN)
                    {
                        num_jpegs_with_unknown_subsampling++;
                    }
                    continue;
                }
                else
                {
                    std::cerr << "ERROR: The chroma sub-sampling is not supported by VCN Hardware"
                              << std::endl;
                    return EXIT_FAILURE;
                }
            }

            // Get channel pitch and sizes
            uint32_t num_channels                         = 0;
            uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
            if(utils.get_channel_pitch_and_sizes(decode_params_batch[index],
                                                 temp_subsampling,
                                                 temp_widths.data(),
                                                 temp_heights.data(),
                                                 num_channels,
                                                 output_images[current_batch_size],
                                                 channel_sizes))
            {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return EXIT_FAILURE;
            }

            // Allocate memory for channels
            for(uint32_t n = 0; n < num_channels; n++)
            {
                if(prior_channel_sizes[current_batch_size][n] != channel_sizes[n])
                {
                    if(output_images[current_batch_size].channel[n] != nullptr)
                    {
                        HIP_CHECK(hipFree(
                            reinterpret_cast<void*>(output_images[current_batch_size].channel[n])));
                        output_images[current_batch_size].channel[n] = nullptr;
                    }
                    HIP_CHECK(
                        hipMalloc(&output_images[current_batch_size].channel[n], channel_sizes[n]));
                    prior_channel_sizes[current_batch_size][n] = channel_sizes[n];
                }
            }

            rocjpeg_stream_handles_for_current_batch[current_batch_size]
                = rocjpeg_stream_handles[index];
            subsamplings[current_batch_size]    = temp_subsampling;
            widths[current_batch_size]          = temp_widths;
            heights[current_batch_size]         = temp_heights;
            base_file_names[current_batch_size] = temp_base_file_name;
            current_batch_size++;
        }

        // Decode batch
        double time_per_batch_in_milli_sec = 0;
        if(current_batch_size > 0)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            ROCJPEG_CHECK(rocJpegDecodeBatched(rocjpeg_handle,
                                               rocjpeg_stream_handles_for_current_batch.data(),
                                               current_batch_size,
                                               decode_params_batch.data(),
                                               output_images.data()));
            auto end_time = std::chrono::high_resolution_clock::now();
            time_per_batch_in_milli_sec
                = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        // Calculate statistics
        double image_size_in_mpixels = 0;
        for(int b = 0; b < current_batch_size; b++)
        {
            image_size_in_mpixels += (static_cast<double>(widths[b][0])
                                      * static_cast<double>(heights[b][0]) / 1000000);
        }

        total_images += current_batch_size;

        // Save images if requested
        if(save_images)
        {
            for(int b = 0; b < current_batch_size; b++)
            {
                std::string image_save_path = output_file_path;
                uint32_t    roi_width       = decode_params_batch[b].crop_rectangle.right
                                     - decode_params_batch[b].crop_rectangle.left;
                uint32_t roi_height = decode_params_batch[b].crop_rectangle.bottom
                                      - decode_params_batch[b].crop_rectangle.top;
                bool is_roi_valid = (roi_width > 0 && roi_height > 0 && roi_width <= widths[b][0]
                                     && roi_height <= heights[b][0]);
                uint32_t width    = is_roi_valid ? roi_width : widths[b][0];
                uint32_t height   = is_roi_valid ? roi_height : heights[b][0];
                if(is_dir)
                {
                    utils.get_output_file_ext(decode_params_batch[b].output_format,
                                              base_file_names[b],
                                              width,
                                              height,
                                              subsamplings[b],
                                              image_save_path);
                }
                utils.save_image(image_save_path,
                                 &output_images[b],
                                 width,
                                 height,
                                 subsamplings[b],
                                 decode_params_batch[b].output_format);
            }
        }

        if(is_dir)
        {
            time_per_image_all += time_per_batch_in_milli_sec;
            mpixels_all += image_size_in_mpixels;
        }
    }

    // Print statistics
    if(is_dir)
    {
        time_per_image_all     = time_per_image_all / total_images;
        double images_per_sec  = 1000 / time_per_image_all;
        double mpixels_per_sec = mpixels_all * images_per_sec / total_images;
        std::cout << "Total decoded images: " << total_images << std::endl;
        if(num_bad_jpegs || num_jpegs_with_411_subsampling || num_jpegs_with_unknown_subsampling
           || num_jpegs_with_unsupported_resolution)
        {
            std::cout << "Total skipped images: "
                      << num_bad_jpegs + num_jpegs_with_411_subsampling
                             + num_jpegs_with_unknown_subsampling
                             + num_jpegs_with_unsupported_resolution;
            if(num_bad_jpegs)
            {
                std::cout << " ,total images that cannot be parsed: " << num_bad_jpegs;
            }
            if(num_jpegs_with_411_subsampling)
            {
                std::cout << " ,total images with YUV 4:1:1 chroma subsampling: "
                          << num_jpegs_with_411_subsampling;
            }
            if(num_jpegs_with_unknown_subsampling)
            {
                std::cout << " ,total images with unknown chroma subsampling: "
                          << num_jpegs_with_unknown_subsampling;
            }
            if(num_jpegs_with_unsupported_resolution)
            {
                std::cout << " ,total images with unsupported resolution: "
                          << num_jpegs_with_unsupported_resolution;
            }
            std::cout << std::endl;
        }
        if(total_images)
        {
            std::cout << "Average processing time per image (ms): " << time_per_image_all
                      << std::endl;
            std::cout << "Average decoded images per sec (Images/Sec): " << images_per_sec
                      << std::endl;
            std::cout << "Average decoded images size (Mpixels/Sec): " << mpixels_per_sec
                      << std::endl;
        }
    }

    // Cleanup
    for(auto& it : output_images)
    {
        for(int i = 0; i < ROCJPEG_MAX_COMPONENT; i++)
        {
            if(it.channel[i] != nullptr)
            {
                HIP_CHECK(hipFree(reinterpret_cast<void*>(it.channel[i])));
                it.channel[i] = nullptr;
            }
        }
    }
    ROCJPEG_CHECK(rocJpegDestroy(rocjpeg_handle));
    for(auto& it : rocjpeg_stream_handles)
    {
        ROCJPEG_CHECK(rocJpegStreamDestroy(it));
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
