// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_ROCJPEG_UTILS_HPP
#define COMMON_ROCJPEG_UTILS_HPP

#include "example_utils.hpp"

#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "rocjpeg/rocjpeg.h"

#define ROCJPEG_CHECK(call)                                                                     \
    {                                                                                           \
        RocJpegStatus rocjpeg_status = (call);                                                  \
        if(rocjpeg_status != ROCJPEG_STATUS_SUCCESS)                                            \
        {                                                                                       \
            std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;                              \
            exit(1);                                                                            \
        }                                                                                       \
    }

/// \brief Utility class for rocJPEG samples.
///
/// This class provides utility functions for rocJPEG samples, such as checking JPEG files,
/// getting file paths, initializing HIP device, getting chroma subsampling string,
/// getting channel pitch and sizes, getting output file extension, and saving images.
class rocjpeg_utils
{
public:
    /// \brief Checks if a file is a JPEG file.
    ///
    /// \param file_path The path to the file to be checked.
    /// \return True if the file is a JPEG file, false otherwise.
    static bool is_jpeg(const std::string& file_path)
    {
        std::ifstream file(file_path, std::ios::binary);
        if(!file.is_open())
        {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return false;
        }

        unsigned char buffer[2];
        file.read(reinterpret_cast<char*>(buffer), 2);
        file.close();

        // The first two bytes of every JPEG stream are always 0xFFD8, which represents the Start of Image (SOI) marker.
        return buffer[0] == 0xFF && buffer[1] == 0xD8;
    }

    /// \brief Gets the file paths.
    ///
    /// This function gets the file paths based on the input path and sets the corresponding variables.
    ///
    /// \param input_path The input path.
    /// \param file_paths The vector to store the file paths.
    /// \param is_dir Flag indicating whether the input path is a directory.
    /// \param is_file Flag indicating whether the input path is a file.
    /// \return True if successful, false otherwise.
    static bool get_file_paths(std::string&              input_path,
                               std::vector<std::string>& file_paths,
                               bool&                     is_dir,
                               bool&                     is_file)
    {
        std::cout << "Reading images from disk, please wait!" << std::endl;
        if(!fs::exists(input_path))
        {
            std::cerr << "ERROR: the input path does not exist!" << std::endl;
            return false;
        }
        is_dir  = fs::is_directory(input_path);
        is_file = fs::is_regular_file(input_path);
        if(is_dir)
        {
            for(const auto& entry : fs::recursive_directory_iterator(input_path))
            {
                if(fs::is_regular_file(entry) && is_jpeg(entry.path().string()))
                {
                    file_paths.push_back(entry.path().string());
                }
            }
        }
        else if(is_file && is_jpeg(input_path))
        {
            file_paths.push_back(input_path);
        }
        else
        {
            std::cerr << "ERROR: the input path does not contain JPEG files!" << std::endl;
            return false;
        }
        return true;
    }

    /// \brief Initializes the HIP device.
    ///
    /// This function initializes the HIP device with the specified device ID.
    ///
    /// \param device_id The device ID.
    /// \return True if successful, false otherwise.
    static bool init_hip_device(int device_id)
    {
        int             num_devices;
        hipDeviceProp_t hip_dev_prop;
        HIP_CHECK(hipGetDeviceCount(&num_devices));
        if(num_devices < 1)
        {
            std::cerr << "ERROR: didn't find any GPU!" << std::endl;
            return false;
        }
        if(device_id >= num_devices)
        {
            std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
            return false;
        }
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipGetDeviceProperties(&hip_dev_prop, device_id));

        std::cout << "Using GPU device " << device_id << ": " << hip_dev_prop.name << "["
                  << hip_dev_prop.gcnArchName << "] on PCI bus " << std::setfill('0')
                  << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":"
                  << std::setfill('0') << std::setw(2) << std::right << std::hex
                  << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec
                  << std::endl;

        return true;
    }

    /// \brief Gets the chroma subsampling string.
    ///
    /// This function gets the chroma subsampling string based on the specified subsampling value.
    ///
    /// \param subsampling The chroma subsampling value.
    /// \param chroma_sub_sampling The string to store the chroma subsampling.
    void get_chroma_subsampling_str(RocJpegChromaSubsampling subsampling,
                                    std::string&             chroma_sub_sampling)
    {
        switch(subsampling)
        {
            case ROCJPEG_CSS_444: chroma_sub_sampling = "YUV 4:4:4"; break;
            case ROCJPEG_CSS_440: chroma_sub_sampling = "YUV 4:4:0"; break;
            case ROCJPEG_CSS_422: chroma_sub_sampling = "YUV 4:2:2"; break;
            case ROCJPEG_CSS_420: chroma_sub_sampling = "YUV 4:2:0"; break;
            case ROCJPEG_CSS_411: chroma_sub_sampling = "YUV 4:1:1"; break;
            case ROCJPEG_CSS_400: chroma_sub_sampling = "YUV 4:0:0"; break;
            case ROCJPEG_CSS_UNKNOWN: chroma_sub_sampling = "UNKNOWN"; break;
            default: chroma_sub_sampling = ""; break;
        }
    }

    /// \brief Gets the channel pitch and sizes.
    ///
    /// This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
    /// output image, and channel sizes.
    ///
    /// \param decode_params The decode parameters that specify the output format and crop rectangle.
    /// \param subsampling The chroma subsampling.
    /// \param widths The array to store the channel widths.
    /// \param heights The array to store the channel heights.
    /// \param num_channels The number of channels.
    /// \param output_image The output image.
    /// \param channel_sizes The array to store the channel sizes.
    /// \return The channel pitch.
    int get_channel_pitch_and_sizes(RocJpegDecodeParams      decode_params,
                                    RocJpegChromaSubsampling subsampling,
                                    uint32_t*                widths,
                                    uint32_t*                heights,
                                    uint32_t&                num_channels,
                                    RocJpegImage&            output_image,
                                    uint32_t*                channel_sizes)
    {

        bool     is_roi_valid = false;
        uint32_t roi_width;
        uint32_t roi_height;
        roi_width  = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
        roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;
        if(roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0])
        {
            is_roi_valid = true;
        }
        switch(decode_params.output_format)
        {
            case ROCJPEG_OUTPUT_NATIVE:
                switch(subsampling)
                {
                    case ROCJPEG_CSS_444:
                        num_channels          = 3;
                        output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0]
                            = is_roi_valid ? align(roi_width, mem_alignment)
                                           : align(widths[0], mem_alignment);
                        channel_sizes[2] = channel_sizes[1] = channel_sizes[0]
                            = output_image.pitch[0]
                              * (is_roi_valid ? align(roi_height, mem_alignment)
                                              : align(heights[0], mem_alignment));
                        break;
                    case ROCJPEG_CSS_440:
                        num_channels          = 3;
                        output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0]
                            = is_roi_valid ? align(roi_width, mem_alignment)
                                           : align(widths[0], mem_alignment);
                        channel_sizes[0] = output_image.pitch[0]
                                           * (is_roi_valid ? align(roi_height, mem_alignment)
                                                           : align(heights[0], mem_alignment));
                        channel_sizes[2] = channel_sizes[1]
                            = output_image.pitch[0]
                              * (is_roi_valid ? align(roi_height >> 1, mem_alignment)
                                              : align(heights[0] >> 1, mem_alignment));
                        break;
                    case ROCJPEG_CSS_422:
                        num_channels          = 1;
                        output_image.pitch[0] = (is_roi_valid ? align(roi_width, mem_alignment)
                                                              : align(widths[0], mem_alignment))
                                                * 2;
                        channel_sizes[0] = output_image.pitch[0]
                                           * (is_roi_valid ? align(roi_height, mem_alignment)
                                                           : align(heights[0], mem_alignment));
                        break;
                    case ROCJPEG_CSS_420:
                        num_channels          = 2;
                        output_image.pitch[1] = output_image.pitch[0]
                            = is_roi_valid ? align(roi_width, mem_alignment)
                                           : align(widths[0], mem_alignment);
                        channel_sizes[0] = output_image.pitch[0]
                                           * (is_roi_valid ? align(roi_height, mem_alignment)
                                                           : align(heights[0], mem_alignment));
                        channel_sizes[1] = output_image.pitch[1]
                                           * (is_roi_valid ? align(roi_height >> 1, mem_alignment)
                                                           : align(heights[0] >> 1, mem_alignment));
                        break;
                    case ROCJPEG_CSS_400:
                        num_channels          = 1;
                        output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment)
                                                             : align(widths[0], mem_alignment);
                        channel_sizes[0]      = output_image.pitch[0]
                                           * (is_roi_valid ? align(roi_height, mem_alignment)
                                                           : align(heights[0], mem_alignment));
                        break;
                    default:
                        std::cout << "Unknown chroma subsampling!" << std::endl;
                        return EXIT_FAILURE;
                }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                if(subsampling == ROCJPEG_CSS_400)
                {
                    num_channels          = 1;
                    output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment)
                                                         : align(widths[0], mem_alignment);
                    channel_sizes[0]      = output_image.pitch[0]
                                       * (is_roi_valid ? align(roi_height, mem_alignment)
                                                       : align(heights[0], mem_alignment));
                }
                else
                {
                    num_channels          = 3;
                    output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment)
                                                         : align(widths[0], mem_alignment);
                    output_image.pitch[1] = is_roi_valid ? align(roi_width, mem_alignment)
                                                         : align(widths[1], mem_alignment);
                    output_image.pitch[2] = is_roi_valid ? align(roi_width, mem_alignment)
                                                         : align(widths[2], mem_alignment);
                    channel_sizes[0]      = output_image.pitch[0]
                                       * (is_roi_valid ? align(roi_height, mem_alignment)
                                                       : align(heights[0], mem_alignment));
                    channel_sizes[1] = output_image.pitch[1]
                                       * (is_roi_valid ? align(roi_height, mem_alignment)
                                                       : align(heights[1], mem_alignment));
                    channel_sizes[2] = output_image.pitch[2]
                                       * (is_roi_valid ? align(roi_height, mem_alignment)
                                                       : align(heights[2], mem_alignment));
                }
                break;
            case ROCJPEG_OUTPUT_Y:
                num_channels          = 1;
                output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment)
                                                     : align(widths[0], mem_alignment);
                channel_sizes[0]      = output_image.pitch[0]
                                   * (is_roi_valid ? align(roi_height, mem_alignment)
                                                   : align(heights[0], mem_alignment));
                break;
            case ROCJPEG_OUTPUT_RGB:
                num_channels          = 1;
                output_image.pitch[0] = (is_roi_valid ? align(roi_width, mem_alignment)
                                                      : align(widths[0], mem_alignment))
                                        * 3;
                channel_sizes[0] = output_image.pitch[0]
                                   * (is_roi_valid ? align(roi_height, mem_alignment)
                                                   : align(heights[0], mem_alignment));
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                num_channels          = 3;
                output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0]
                    = is_roi_valid ? align(roi_width, mem_alignment)
                                   : align(widths[0], mem_alignment);
                channel_sizes[2] = channel_sizes[1] = channel_sizes[0]
                    = output_image.pitch[0]
                      * (is_roi_valid ? align(roi_height, mem_alignment)
                                      : align(heights[0], mem_alignment));
                break;
            default: std::cout << "Unknown output format!" << std::endl; return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    /// \brief Gets the output file extension.
    ///
    /// This function gets the output file extension based on the specified output format, base file name,
    /// image width, image height, and file name for saving.
    ///
    /// \param output_format The output format.
    /// \param base_file_name The base file name.
    /// \param image_width The image width.
    /// \param image_height The image height.
    /// \param subsampling The chroma subsampling.
    /// \param file_name_for_saving The string to store the file name for saving.
    void get_output_file_ext(RocJpegOutputFormat      output_format,
                             std::string&             base_file_name,
                             uint32_t                 image_width,
                             uint32_t                 image_height,
                             RocJpegChromaSubsampling subsampling,
                             std::string&             file_name_for_saving)
    {
        std::string            file_extension;
        std::string::size_type p(base_file_name.find_last_of('.'));
        std::string            file_name_no_ext   = base_file_name.substr(0, p);
        std::string            format_description = "";
        switch(output_format)
        {
            case ROCJPEG_OUTPUT_NATIVE:
                file_extension = "yuv";
                switch(subsampling)
                {
                    case ROCJPEG_CSS_444: format_description = "444"; break;
                    case ROCJPEG_CSS_440: format_description = "440"; break;
                    case ROCJPEG_CSS_422: format_description = "422_yuyv"; break;
                    case ROCJPEG_CSS_420: format_description = "nv12"; break;
                    case ROCJPEG_CSS_400: format_description = "400"; break;
                    default: std::cout << "Unknown chroma subsampling!" << std::endl; return;
                }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                file_extension     = "yuv";
                format_description = "planar";
                break;
            case ROCJPEG_OUTPUT_Y:
                file_extension     = "yuv";
                format_description = "400";
                break;
            case ROCJPEG_OUTPUT_RGB:
                file_extension     = "rgb";
                format_description = "packed";
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                file_extension     = "rgb";
                format_description = "planar";
                break;
            default: file_extension = ""; break;
        }
        file_name_for_saving += "//" + file_name_no_ext + "_" + std::to_string(image_width) + "x"
                                + std::to_string(image_height) + "_" + format_description + "."
                                + file_extension;
    }

    /// \brief Saves the image.
    ///
    /// This function saves the image to the specified output file name based on the output image, image width,
    /// image height, chroma subsampling, and output format.
    ///
    /// \param output_file_name The output file name.
    /// \param output_image The output image.
    /// \param img_width The image width.
    /// \param img_height The image height.
    /// \param subsampling The chroma subsampling.
    /// \param output_format The output format.
    void save_image(std::string              output_file_name,
                    RocJpegImage*            output_image,
                    uint32_t                 img_width,
                    uint32_t                 img_height,
                    RocJpegChromaSubsampling subsampling,
                    RocJpegOutputFormat      output_format)
    {
        uint8_t* hst_ptr = nullptr;
        FILE*    fp;

        if(output_image == nullptr || output_image->channel[0] == nullptr
           || output_image->pitch[0] == 0)
        {
            return;
        }

        uint32_t widths[ROCJPEG_MAX_COMPONENT]         = {};
        uint32_t heights[ROCJPEG_MAX_COMPONENT]        = {};
        uint32_t aliged_heights[ROCJPEG_MAX_COMPONENT] = {};

        switch(output_format)
        {
            case ROCJPEG_OUTPUT_NATIVE:
                switch(subsampling)
                {
                    case ROCJPEG_CSS_444:
                        widths[2] = widths[1] = widths[0] = img_width;
                        heights[2] = heights[1] = heights[0] = img_height;
                        break;
                    case ROCJPEG_CSS_440:
                        widths[2] = widths[1] = widths[0] = img_width;
                        heights[0]                        = img_height;
                        heights[2] = heights[1] = img_height >> 1;
                        break;
                    case ROCJPEG_CSS_422:
                        widths[0]  = img_width * 2;
                        heights[0] = img_height;
                        break;
                    case ROCJPEG_CSS_420:
                        widths[1] = widths[0] = img_width;
                        heights[0]            = img_height;
                        heights[1]            = img_height >> 1;
                        break;
                    case ROCJPEG_CSS_400:
                        widths[0]  = img_width;
                        heights[0] = img_height;
                        break;
                    default: std::cout << "Unknown chroma subsampling!" << std::endl; return;
                }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                switch(subsampling)
                {
                    case ROCJPEG_CSS_444:
                        widths[2] = widths[1] = widths[0] = img_width;
                        heights[2] = heights[1] = heights[0] = img_height;
                        break;
                    case ROCJPEG_CSS_440:
                        widths[2] = widths[1] = widths[0] = img_width;
                        heights[0]                        = img_height;
                        heights[2] = heights[1] = img_height >> 1;
                        break;
                    case ROCJPEG_CSS_422:
                        widths[0] = img_width;
                        widths[2] = widths[1] = widths[0] >> 1;
                        heights[2] = heights[1] = heights[0] = img_height;
                        break;
                    case ROCJPEG_CSS_420:
                        widths[0] = img_width;
                        widths[2] = widths[1] = widths[0] >> 1;
                        heights[0]            = img_height;
                        heights[2] = heights[1] = img_height >> 1;
                        break;
                    case ROCJPEG_CSS_400:
                        widths[0]  = img_width;
                        heights[0] = img_height;
                        break;
                    default: std::cout << "Unknown chroma subsampling!" << std::endl; return;
                }
                break;
            case ROCJPEG_OUTPUT_Y:
                widths[0]  = img_width;
                heights[0] = img_height;
                break;
            case ROCJPEG_OUTPUT_RGB:
                widths[0]  = img_width * 3;
                heights[0] = img_height;
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                widths[2] = widths[1] = widths[0] = img_width;
                heights[2] = heights[1] = heights[0] = img_height;
                break;
            default: std::cout << "Unknown output format!" << std::endl; return;
        }

        aliged_heights[0]      = align(heights[0], mem_alignment);
        aliged_heights[1]      = align(heights[1], mem_alignment);
        aliged_heights[2]      = align(heights[2], mem_alignment);
        uint32_t channel0_size = output_image->pitch[0] * aliged_heights[0];
        uint32_t channel1_size = output_image->pitch[1] * aliged_heights[1];
        uint32_t channel2_size = output_image->pitch[2] * aliged_heights[2];

        uint32_t output_image_size = channel0_size + channel1_size + channel2_size;

        if(hst_ptr == nullptr)
        {
            hst_ptr = new uint8_t[output_image_size];
        }

        HIP_CHECK(hipMemcpyDtoH((void*)hst_ptr, output_image->channel[0], channel0_size));

        uint8_t* tmp_hst_ptr = hst_ptr;
        fp                   = fopen(output_file_name.c_str(), "wb");
        if(fp)
        {
            // write channel0
            if(widths[0] == output_image->pitch[0] && heights[0] == aliged_heights[0])
            {
                fwrite(hst_ptr, 1, channel0_size, fp);
            }
            else
            {
                for(uint32_t i = 0; i < heights[0]; i++)
                {
                    fwrite(tmp_hst_ptr, 1, widths[0], fp);
                    tmp_hst_ptr += output_image->pitch[0];
                }
            }
            // write channel1
            if(channel1_size != 0 && output_image->channel[1] != nullptr)
            {
                uint8_t* channel1_hst_ptr = hst_ptr + channel0_size;
                HIP_CHECK(hipMemcpyDtoH((void*)channel1_hst_ptr,
                                        output_image->channel[1],
                                        channel1_size));
                if(widths[1] == output_image->pitch[1] && heights[1] == aliged_heights[1])
                {
                    fwrite(channel1_hst_ptr, 1, channel1_size, fp);
                }
                else
                {
                    for(uint32_t i = 0; i < heights[1]; i++)
                    {
                        fwrite(channel1_hst_ptr, 1, widths[1], fp);
                        channel1_hst_ptr += output_image->pitch[1];
                    }
                }
            }
            // write channel2
            if(channel2_size != 0 && output_image->channel[2] != nullptr)
            {
                uint8_t* channel2_hst_ptr = hst_ptr + channel0_size + channel1_size;
                HIP_CHECK(hipMemcpyDtoH((void*)channel2_hst_ptr,
                                        output_image->channel[2],
                                        channel2_size));
                if(widths[2] == output_image->pitch[2] && heights[2] == aliged_heights[2])
                {
                    fwrite(channel2_hst_ptr, 1, channel2_size, fp);
                }
                else
                {
                    for(uint32_t i = 0; i < heights[2]; i++)
                    {
                        fwrite(channel2_hst_ptr, 1, widths[2], fp);
                        channel2_hst_ptr += output_image->pitch[2];
                    }
                }
            }
            fclose(fp);
        }

        if(hst_ptr != nullptr)
        {
            delete[] hst_ptr;
            hst_ptr     = nullptr;
            tmp_hst_ptr = nullptr;
        }
    }

private:
    static const int mem_alignment = 16;

    /// \brief Aligns a value to a specified alignment.
    ///
    /// This function takes a value and aligns it to the specified alignment. It returns the aligned value.
    ///
    /// \param value The value to be aligned.
    /// \param alignment The alignment value.
    /// \return The aligned value.
    static inline int align(int value, int alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }
};

/// \brief Thread pool for parallel JPEG decoding.
class thread_pool
{
public:
    thread_pool(int nthreads) : shutdown_(false)
    {
        // Create the specified number of threads
        threads_.reserve(nthreads);
        for(int i = 0; i < nthreads; ++i)
        {
            threads_.emplace_back(std::bind(&thread_pool::thread_entry, this, i));
        }
    }

    ~thread_pool() {}

    void join_threads()
    {
        {
            // Unblock any threads and tell them to stop
            std::unique_lock<std::mutex> lock(mutex_);
            shutdown_ = true;
            cond_var_.notify_all();
        }

        // Wait for all threads to stop
        for(auto& thread : threads_)
        {
            thread.join();
        }
    }

    void execute_job(std::function<void()> func)
    {
        // Place a job on the queue and unblock a thread
        std::unique_lock<std::mutex> lock(mutex_);
        decode_jobs_queue_.emplace(std::move(func));
        cond_var_.notify_one();
    }

protected:
    void thread_entry(int /* i */)
    {
        std::function<void()> execute_decode_job;

        while(true)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [&] { return shutdown_ || !decode_jobs_queue_.empty(); });
                if(decode_jobs_queue_.empty())
                {
                    // No jobs to do; shutting down
                    return;
                }

                execute_decode_job = std::move(decode_jobs_queue_.front());
                decode_jobs_queue_.pop();
            }

            // Execute the decode job without holding any locks
            execute_decode_job();
        }
    }

    std::mutex                        mutex_;
    std::condition_variable           cond_var_;
    bool                              shutdown_;
    std::queue<std::function<void()>> decode_jobs_queue_;
    std::vector<std::thread>          threads_;
};

#endif // COMMON_ROCJPEG_UTILS_HPP
