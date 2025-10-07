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
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct decode_info
{
    std::vector<std::string>         file_paths;
    RocJpegHandle                    rocjpeg_handle;
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles;
    uint64_t                         num_decoded_images;
    double                           images_per_sec;
    double                           image_size_in_mpixels_per_sec;
    uint64_t                         num_bad_jpegs;
    uint64_t                         num_jpegs_with_411_subsampling;
    uint64_t                         num_jpegs_with_unknown_subsampling;
    uint64_t                         num_jpegs_with_unsupported_resolution;
};

void decode_images(decode_info&         decode_info_ref,
                   rocjpeg_utils        rocjpeg_utils_obj,
                   RocJpegDecodeParams& decode_params,
                   bool                 save_images,
                   std::string&         output_file_path,
                   int                  batch_size,
                   int                  device_id)
{
    bool                               is_roi_valid = false;
    uint32_t                           roi_width;
    uint32_t                           roi_height;
    uint8_t                            num_components;
    uint32_t                           channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::string                        chroma_sub_sampling                  = "";
    uint32_t                           num_channels                         = 0;
    double                             image_size_in_mpixels_all            = 0;
    double                             total_decode_time_in_milli_sec       = 0;
    int                                current_batch_size                   = 0;
    std::vector<std::vector<char>>     batch_images(batch_size);
    std::vector<std::vector<uint32_t>> widths(batch_size,
                                              std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>> heights(batch_size,
                                               std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>> prior_channel_sizes(
        batch_size,
        std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<RocJpegChromaSubsampling> subsamplings(batch_size);
    std::vector<RocJpegImage>             output_images(batch_size);
    std::vector<RocJpegDecodeParams>      decode_params_batch(batch_size, decode_params);
    std::vector<std::string>              base_file_names(batch_size);
    std::vector<RocJpegStreamHandle>      rocjpeg_stream_handles(batch_size);
    std::vector<uint32_t>                 temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t>                 temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling              temp_subsampling;
    std::string                           temp_base_file_name;

    HIP_CHECK(hipSetDevice(device_id));

    for(int i = 0; i < static_cast<int>(decode_info_ref.file_paths.size()); i += batch_size)
    {
        int batch_end
            = std::min(i + batch_size, static_cast<int>(decode_info_ref.file_paths.size()));
        for(int j = i; j < batch_end; j++)
        {
            int index = j - i;

            temp_base_file_name = decode_info_ref.file_paths[j].substr(
                decode_info_ref.file_paths[j].find_last_of("/\\") + 1);

            // Read image from disk
            std::ifstream input(decode_info_ref.file_paths[j].c_str(),
                                std::ios::in | std::ios::binary | std::ios::ate);
            if(!input.is_open())
            {
                std::cerr << "ERROR: Cannot open image: " << decode_info_ref.file_paths[j]
                          << std::endl;
                return;
            }

            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);

            if(batch_images[index].size() < static_cast<size_t>(file_size))
            {
                batch_images[index].resize(file_size);
            }

            if(!input.read(batch_images[index].data(), file_size))
            {
                std::cerr << "ERROR: Cannot read from file: " << decode_info_ref.file_paths[j]
                          << std::endl;
                return;
            }

            // Parse JPEG stream
            RocJpegStatus rocjpeg_status
                = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_images[index].data()),
                                     file_size,
                                     decode_info_ref.rocjpeg_stream_handles[index]);
            if(rocjpeg_status != ROCJPEG_STATUS_SUCCESS)
            {
                decode_info_ref.num_bad_jpegs++;
                std::cerr << "Skipping decoding input file: " << decode_info_ref.file_paths[j]
                          << std::endl;
                continue;
            }

            ROCJPEG_CHECK(rocJpegGetImageInfo(decode_info_ref.rocjpeg_handle,
                                              decode_info_ref.rocjpeg_stream_handles[index],
                                              &num_components,
                                              &temp_subsampling,
                                              temp_widths.data(),
                                              temp_heights.data()));

            rocjpeg_utils_obj.get_chroma_subsampling_str(temp_subsampling, chroma_sub_sampling);

            if(temp_widths[0] < 64 || temp_heights[0] < 64)
            {
                decode_info_ref.num_jpegs_with_unsupported_resolution++;
                continue;
            }

            if(temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN)
            {
                if(temp_subsampling == ROCJPEG_CSS_411)
                {
                    decode_info_ref.num_jpegs_with_411_subsampling++;
                }
                if(temp_subsampling == ROCJPEG_CSS_UNKNOWN)
                {
                    decode_info_ref.num_jpegs_with_unknown_subsampling++;
                }
                continue;
            }

            if(rocjpeg_utils_obj.get_channel_pitch_and_sizes(decode_params_batch[index],
                                                             temp_subsampling,
                                                             temp_widths.data(),
                                                             temp_heights.data(),
                                                             num_channels,
                                                             output_images[current_batch_size],
                                                             channel_sizes))
            {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return;
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

            rocjpeg_stream_handles[current_batch_size]
                = decode_info_ref.rocjpeg_stream_handles[index];
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
            ROCJPEG_CHECK(rocJpegDecodeBatched(decode_info_ref.rocjpeg_handle,
                                               rocjpeg_stream_handles.data(),
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

        decode_info_ref.num_decoded_images += current_batch_size;

        // Save images if requested
        if(save_images)
        {
            for(int b = 0; b < current_batch_size; b++)
            {
                std::string image_save_path = output_file_path;
                roi_width                   = decode_params_batch[b].crop_rectangle.right
                            - decode_params_batch[b].crop_rectangle.left;
                roi_height = decode_params_batch[b].crop_rectangle.bottom
                             - decode_params_batch[b].crop_rectangle.top;
                is_roi_valid    = (roi_width > 0 && roi_height > 0 && roi_width <= widths[b][0]
                                && roi_height <= heights[b][0]);
                uint32_t width  = is_roi_valid ? roi_width : widths[b][0];
                uint32_t height = is_roi_valid ? roi_height : heights[b][0];
                rocjpeg_utils_obj.get_output_file_ext(decode_params.output_format,
                                                      base_file_names[b],
                                                      width,
                                                      height,
                                                      subsamplings[b],
                                                      image_save_path);
                rocjpeg_utils_obj.save_image(image_save_path,
                                             &output_images[b],
                                             width,
                                             height,
                                             subsamplings[b],
                                             decode_params.output_format);
            }
        }

        total_decode_time_in_milli_sec += time_per_batch_in_milli_sec;
        image_size_in_mpixels_all += image_size_in_mpixels;

        current_batch_size = 0;
    }

    double avg_time_per_image
        = decode_info_ref.num_decoded_images > 0
              ? total_decode_time_in_milli_sec / decode_info_ref.num_decoded_images
              : 0;
    decode_info_ref.images_per_sec = avg_time_per_image > 0 ? 1000 / avg_time_per_image : 0;
    decode_info_ref.image_size_in_mpixels_per_sec = decode_info_ref.num_decoded_images > 0
                                                        ? decode_info_ref.images_per_sec
                                                              * image_size_in_mpixels_all
                                                              / decode_info_ref.num_decoded_images
                                                        : 0;

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
}

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
    parser.set_optional<int>("threads",
                             "threads",
                             1,
                             "Number of threads (<= 32) for parallel JPEG decoding");
    parser.set_optional<int>("batch_size", "batch_size", 1, "Batch size for decoding");
    parser.set_optional<std::string>("crop", "crop", "", "Crop rectangle: left,top,right,bottom");
    parser.run_and_exit_if_error();

    // Get parsed arguments
    std::string input_path       = parser.get<std::string>("input");
    std::string output_file_path = parser.get<std::string>("output");
    bool        save_images      = !output_file_path.empty();
    int         device_id        = parser.get<int>("device");
    int         backend_int      = parser.get<int>("backend");
    std::string format_str       = parser.get<std::string>("format");
    int         num_threads      = parser.get<int>("threads");
    int         batch_size       = parser.get<int>("batch_size");
    std::string crop_str         = parser.get<std::string>("crop");

    // Validate num_threads
    if(num_threads <= 0 || num_threads > 32)
    {
        std::cerr << "ERROR: Number of threads must be between 1 and 32" << std::endl;
        return EXIT_FAILURE;
    }

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

    // Adjust num_threads based on file count
    if(num_threads > static_cast<int>(file_paths.size()))
    {
        num_threads = file_paths.size();
    }

    // Initialize decode info for each thread
    std::vector<decode_info> decode_info_per_thread(num_threads);

    for(int i = 0; i < num_threads; i++)
    {
        ROCJPEG_CHECK(
            rocJpegCreate(rocjpeg_backend, device_id, &decode_info_per_thread[i].rocjpeg_handle));
        decode_info_per_thread[i].rocjpeg_stream_handles.resize(batch_size);
        for(int j = 0; j < batch_size; j++)
        {
            ROCJPEG_CHECK(
                rocJpegStreamCreate(&decode_info_per_thread[i].rocjpeg_stream_handles[j]));
        }
        decode_info_per_thread[i].num_decoded_images                    = 0;
        decode_info_per_thread[i].images_per_sec                        = 0;
        decode_info_per_thread[i].image_size_in_mpixels_per_sec         = 0;
        decode_info_per_thread[i].num_bad_jpegs                         = 0;
        decode_info_per_thread[i].num_jpegs_with_411_subsampling        = 0;
        decode_info_per_thread[i].num_jpegs_with_unknown_subsampling    = 0;
        decode_info_per_thread[i].num_jpegs_with_unsupported_resolution = 0;
    }

    // Create thread pool
    thread_pool pool(num_threads);

    // Distribute files among threads
    size_t files_per_thread = file_paths.size() / num_threads;
    size_t remaining_files  = file_paths.size() % num_threads;
    size_t start_index      = 0;
    for(int i = 0; i < num_threads; i++)
    {
        size_t end_index
            = start_index + files_per_thread + (i < static_cast<int>(remaining_files) ? 1 : 0);
        decode_info_per_thread[i].file_paths.assign(file_paths.begin() + start_index,
                                                    file_paths.begin() + end_index);
        start_index = end_index;
    }

    rocjpeg_utils utils;
    std::cout << "Decoding started with " << num_threads << " threads, please wait!" << std::endl;

    // Execute decoding jobs
    for(int i = 0; i < num_threads; ++i)
    {
        pool.execute_job(std::bind(decode_images,
                                   std::ref(decode_info_per_thread[i]),
                                   utils,
                                   std::ref(decode_params),
                                   save_images,
                                   std::ref(output_file_path),
                                   batch_size,
                                   device_id));
    }
    pool.join_threads();

    // Aggregate statistics
    uint64_t total_decoded_images                        = 0;
    double   total_images_per_sec                        = 0;
    double   total_image_size_in_mpixels_per_sec         = 0;
    uint64_t total_num_bad_jpegs                         = 0;
    uint64_t total_num_jpegs_with_411_subsampling        = 0;
    uint64_t total_num_jpegs_with_unknown_subsampling    = 0;
    uint64_t total_num_jpegs_with_unsupported_resolution = 0;

    for(int i = 0; i < num_threads; i++)
    {
        total_decoded_images += decode_info_per_thread[i].num_decoded_images;
        total_image_size_in_mpixels_per_sec
            += decode_info_per_thread[i].image_size_in_mpixels_per_sec;
        total_images_per_sec += decode_info_per_thread[i].images_per_sec;
        total_num_bad_jpegs += decode_info_per_thread[i].num_bad_jpegs;
        total_num_jpegs_with_411_subsampling
            += decode_info_per_thread[i].num_jpegs_with_411_subsampling;
        total_num_jpegs_with_unknown_subsampling
            += decode_info_per_thread[i].num_jpegs_with_unknown_subsampling;
        total_num_jpegs_with_unsupported_resolution
            += decode_info_per_thread[i].num_jpegs_with_unsupported_resolution;
    }

    // Print statistics
    std::cout << "Total decoded images: " << total_decoded_images << std::endl;
    if(total_num_bad_jpegs || total_num_jpegs_with_411_subsampling
       || total_num_jpegs_with_unknown_subsampling || total_num_jpegs_with_unsupported_resolution)
    {
        std::cout << "Total skipped images: "
                  << total_num_bad_jpegs + total_num_jpegs_with_411_subsampling
                         + total_num_jpegs_with_unknown_subsampling
                         + total_num_jpegs_with_unsupported_resolution;
        if(total_num_bad_jpegs)
        {
            std::cout << " ,total images that cannot be parsed: " << total_num_bad_jpegs;
        }
        if(total_num_jpegs_with_411_subsampling)
        {
            std::cout << " ,total images with YUV 4:1:1 chroma subsampling: "
                      << total_num_jpegs_with_411_subsampling;
        }
        if(total_num_jpegs_with_unknown_subsampling)
        {
            std::cout << " ,total images with unknown chroma subsampling: "
                      << total_num_jpegs_with_unknown_subsampling;
        }
        if(total_num_jpegs_with_unsupported_resolution)
        {
            std::cout << " ,total images with unsupported resolution: "
                      << total_num_jpegs_with_unsupported_resolution;
        }
        std::cout << std::endl;
    }

    if(total_decoded_images > 0)
    {
        std::cout << "Average processing time per image (ms): " << 1000 / total_images_per_sec
                  << std::endl;
        std::cout << "Average decoded images per sec (Images/Sec): " << total_images_per_sec
                  << std::endl;
        std::cout << "Average decoded images size (Mpixels/Sec): "
                  << total_image_size_in_mpixels_per_sec << std::endl;
    }

    // Cleanup
    for(int i = 0; i < num_threads; i++)
    {
        ROCJPEG_CHECK(rocJpegDestroy(decode_info_per_thread[i].rocjpeg_handle));
        for(int j = 0; j < batch_size; j++)
        {
            ROCJPEG_CHECK(
                rocJpegStreamDestroy(decode_info_per_thread[i].rocjpeg_stream_handles[j]));
        }
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
