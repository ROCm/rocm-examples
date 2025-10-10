// MIT License
//
// Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_RPP_UTILS_HPP
#define COMMON_RPP_UTILS_HPP

#include "example_utils.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rpp.h>

/// \brief Overloaded check functions for different RPP status types
inline void check_rpp_status(rppStatus_t status, const char* file, int line)
{
    if(status != rppStatusSuccess)
    {
        std::cerr << "RPP error encountered: " << status << " at " << file << ':' << line
                  << std::endl;
        std::exit(error_exit_code);
    }
}

inline void check_rpp_status(RppStatus status, const char* file, int line)
{
    if(status != RPP_SUCCESS)
    {
        std::cerr << "RPP error encountered: " << status << " at " << file << ':' << line
                  << std::endl;
        std::exit(error_exit_code);
    }
}

/// \brief Checks if the provided RPP status is RPP_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code. Works with both RppStatus and rppStatus_t return types.
#define RPP_CHECK(condition) check_rpp_status(condition, __FILE__, __LINE__)

/// \brief Get the size in bytes of an RPP data type
inline size_t get_size_of_data_type(RpptDataType data_type)
{
    if(data_type == RpptDataType::U8)
    {
        return sizeof(Rpp8u);
    }
    else if(data_type == RpptDataType::I8)
    {
        return sizeof(Rpp8s);
    }
    else if(data_type == RpptDataType::F16)
    {
        return sizeof(Rpp16f);
    }
    else if(data_type == RpptDataType::F32)
    {
        return sizeof(Rpp32f);
    }
    else
    {
        return 0;
    }
}

/// \brief Determine the number of input channels based on layout type
inline int set_input_channels(int layout_type)
{
    if(layout_type == 0 || layout_type == 1)
    {
        return 3;
    }
    else
    {
        return 1;
    }
}

/// \brief Set the layout types in source and destination descriptors
inline void set_descriptor_layout(RpptDescPtr src_desc_ptr,
                                  RpptDescPtr dst_desc_ptr,
                                  int         layout_type,
                                  int         output_format_toggle)
{
    if(layout_type == 0)
    {
        src_desc_ptr->layout = RpptLayout::NHWC;
        if(output_format_toggle == 0)
        {
            dst_desc_ptr->layout = RpptLayout::NHWC;
        }
        else if(output_format_toggle == 1)
        {
            dst_desc_ptr->layout = RpptLayout::NCHW;
        }
    }
    else if(layout_type == 1)
    {
        src_desc_ptr->layout = RpptLayout::NCHW;
        if(output_format_toggle == 0)
        {
            dst_desc_ptr->layout = RpptLayout::NCHW;
        }
        else if(output_format_toggle == 1)
        {
            dst_desc_ptr->layout = RpptLayout::NHWC;
        }
    }
    else
    {
        src_desc_ptr->layout = RpptLayout::NCHW;
        dst_desc_ptr->layout = RpptLayout::NCHW;
    }
}

/// \brief Set the data types in source and destination descriptors
inline void
    set_descriptor_data_type(int bit_depth, RpptDescPtr src_desc_ptr, RpptDescPtr dst_desc_ptr)
{
    if(bit_depth == 0)
    {
        src_desc_ptr->dataType = RpptDataType::U8;
        dst_desc_ptr->dataType = RpptDataType::U8;
    }
    else if(bit_depth == 1)
    {
        src_desc_ptr->dataType = RpptDataType::F16;
        dst_desc_ptr->dataType = RpptDataType::F16;
    }
    else if(bit_depth == 2)
    {
        src_desc_ptr->dataType = RpptDataType::F32;
        dst_desc_ptr->dataType = RpptDataType::F32;
    }
    else if(bit_depth == 3)
    {
        src_desc_ptr->dataType = RpptDataType::U8;
        dst_desc_ptr->dataType = RpptDataType::F16;
    }
    else if(bit_depth == 4)
    {
        src_desc_ptr->dataType = RpptDataType::U8;
        dst_desc_ptr->dataType = RpptDataType::F32;
    }
    else if(bit_depth == 5)
    {
        src_desc_ptr->dataType = RpptDataType::I8;
        dst_desc_ptr->dataType = RpptDataType::I8;
    }
    else if(bit_depth == 6)
    {
        src_desc_ptr->dataType = RpptDataType::U8;
        dst_desc_ptr->dataType = RpptDataType::I8;
    }
}

/// \brief Set dimensions and strides in a descriptor
inline void set_descriptor_dims_and_strides(RpptDescPtr desc_ptr,
                                            int         num_images,
                                            int         max_height,
                                            int         max_width,
                                            int         num_channels,
                                            int         offset_in_bytes)
{
    desc_ptr->numDims       = 4;
    desc_ptr->offsetInBytes = offset_in_bytes;
    desc_ptr->n             = num_images;
    desc_ptr->h             = max_height;
    desc_ptr->w             = max_width;
    desc_ptr->c             = num_channels;

    // Optionally set w stride as a multiple of 8 for src/dst
    desc_ptr->w = ((desc_ptr->w / 8) * 8) + 8;

    // Set strides
    if(desc_ptr->layout == RpptLayout::NHWC)
    {
        desc_ptr->strides.nStride = desc_ptr->c * desc_ptr->w * desc_ptr->h;
        desc_ptr->strides.hStride = desc_ptr->c * desc_ptr->w;
        desc_ptr->strides.wStride = desc_ptr->c;
        desc_ptr->strides.cStride = 1;
    }
    else if(desc_ptr->layout == RpptLayout::NCHW)
    {
        desc_ptr->strides.nStride = desc_ptr->c * desc_ptr->w * desc_ptr->h;
        desc_ptr->strides.cStride = desc_ptr->w * desc_ptr->h;
        desc_ptr->strides.hStride = desc_ptr->w;
        desc_ptr->strides.wStride = 1;
    }
}

/// \brief Read a batch of images using OpenCV
inline void read_image_batch_opencv(Rpp8u*                          input,
                                    RpptDescPtr                     desc_ptr,
                                    const std::vector<std::string>& image_paths)
{
    for(Rpp32u i = 0; i < desc_ptr->n; i++)
    {
        Rpp8u*      input_temp       = input + (i * desc_ptr->strides.nStride);
        std::string input_image_path = image_paths[i];
        cv::Mat     image, image_bgr;

        if(desc_ptr->c == 3)
        {
            image_bgr = cv::imread(input_image_path, 1);
            cv::cvtColor(image_bgr, image, cv::COLOR_BGR2RGB);
        }
        else if(desc_ptr->c == 1)
        {
            image = cv::imread(input_image_path, 0);
        }

        int    width           = image.cols;
        int    height          = image.rows;
        Rpp32u elements_in_row = width * desc_ptr->c;
        Rpp8u* input_image     = image.data;

        for(int j = 0; j < height; j++)
        {
            std::memcpy(input_temp, input_image, elements_in_row * sizeof(Rpp8u));
            input_image += elements_in_row;
            input_temp += desc_ptr->w * desc_ptr->c;
        }
    }
}

/// \brief Write a batch of images using OpenCV
inline void write_image_batch_opencv(const std::string&              output_folder,
                                     Rpp8u*                          output,
                                     RpptDescPtr                     dst_desc_ptr,
                                     const std::vector<std::string>& image_names,
                                     RpptImagePatch*                 dst_img_sizes)
{
    // Create output folder
    mkdir(output_folder.c_str(), 0700);
    std::string output_folder_path = output_folder + "/";

    Rpp32u elements_in_row_max = dst_desc_ptr->w * dst_desc_ptr->c;
    Rpp8u* offsetted_output    = output + dst_desc_ptr->offsetInBytes;

    for(Rpp32u j = 0; j < dst_desc_ptr->n; j++)
    {
        Rpp32u height          = dst_img_sizes[j].height;
        Rpp32u width           = dst_img_sizes[j].width;
        Rpp32u elements_in_row = width * dst_desc_ptr->c;
        Rpp32u output_size     = height * width * dst_desc_ptr->c;

        Rpp8u* temp_output     = static_cast<Rpp8u*>(calloc(output_size, sizeof(Rpp8u)));
        Rpp8u* temp_output_row = temp_output;
        Rpp8u* output_row      = offsetted_output + j * dst_desc_ptr->strides.nStride;

        for(int k = 0; k < static_cast<int>(height); k++)
        {
            std::memcpy(temp_output_row, output_row, elements_in_row * sizeof(Rpp8u));
            temp_output_row += elements_in_row;
            output_row += elements_in_row_max;
        }

        std::string output_image_path = output_folder_path + image_names[j];
        cv::Mat     mat_output_image, mat_output_image_rgb;

        if(dst_desc_ptr->c == 1)
        {
            mat_output_image = cv::Mat(height, width, CV_8UC1, temp_output);
        }
        else if(dst_desc_ptr->c == 2)
        {
            mat_output_image = cv::Mat(height, width, CV_8UC2, temp_output);
        }
        else if(dst_desc_ptr->c == 3)
        {
            mat_output_image_rgb = cv::Mat(height, width, CV_8UC3, temp_output);
            cv::cvtColor(mat_output_image_rgb, mat_output_image, cv::COLOR_RGB2BGR);
        }

        cv::imwrite(output_image_path, mat_output_image);
        free(temp_output);
    }
}

/// \brief Convert PKD3 layout to PLN3 layout
inline void convert_pkd3_to_pln3(Rpp8u* input, RpptDescPtr desc_ptr)
{
    unsigned long long buffer_size = (static_cast<unsigned long long>(desc_ptr->h)
                                      * static_cast<unsigned long long>(desc_ptr->w)
                                      * static_cast<unsigned long long>(desc_ptr->c)
                                      * static_cast<unsigned long long>(desc_ptr->n))
                                     + desc_ptr->offsetInBytes;

    Rpp8u* input_copy = static_cast<Rpp8u*>(calloc(buffer_size, sizeof(Rpp8u)));
    std::memcpy(input_copy, input, buffer_size * sizeof(Rpp8u));

    Rpp8u* input_temp = input + desc_ptr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(desc_ptr->n)
    for(Rpp32u count = 0; count < desc_ptr->n; count++)
    {
        Rpp8u* input_temp_r = input_temp + count * desc_ptr->strides.nStride;
        Rpp8u* input_temp_g = input_temp_r + desc_ptr->strides.cStride;
        Rpp8u* input_temp_b = input_temp_g + desc_ptr->strides.cStride;
        Rpp8u* input_copy_temp
            = input_copy + desc_ptr->offsetInBytes + count * desc_ptr->strides.nStride;

        for(Rpp32u i = 0; i < desc_ptr->h; i++)
        {
            for(Rpp32u j = 0; j < desc_ptr->w; j++)
            {
                *input_temp_r = *input_copy_temp;
                input_copy_temp++;
                input_temp_r++;
                *input_temp_g = *input_copy_temp;
                input_copy_temp++;
                input_temp_g++;
                *input_temp_b = *input_copy_temp;
                input_copy_temp++;
                input_temp_b++;
            }
        }
    }

    free(input_copy);
}

/// \brief Convert PLN3 layout to PKD3 layout
inline void convert_pln3_to_pkd3(Rpp8u* output, RpptDescPtr desc_ptr)
{
    unsigned long long buffer_size = (static_cast<unsigned long long>(desc_ptr->h)
                                      * static_cast<unsigned long long>(desc_ptr->w)
                                      * static_cast<unsigned long long>(desc_ptr->c)
                                      * static_cast<unsigned long long>(desc_ptr->n))
                                     + desc_ptr->offsetInBytes;

    Rpp8u* output_copy = static_cast<Rpp8u*>(calloc(buffer_size, sizeof(Rpp8u)));
    std::memcpy(output_copy, output, buffer_size * sizeof(Rpp8u));

    Rpp8u* output_copy_temp = output_copy + desc_ptr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(desc_ptr->n)
    for(Rpp32u count = 0; count < desc_ptr->n; count++)
    {
        Rpp8u* output_copy_temp_r = output_copy_temp + count * desc_ptr->strides.nStride;
        Rpp8u* output_copy_temp_g = output_copy_temp_r + desc_ptr->strides.cStride;
        Rpp8u* output_copy_temp_b = output_copy_temp_g + desc_ptr->strides.cStride;
        Rpp8u* output_temp = output + desc_ptr->offsetInBytes + count * desc_ptr->strides.nStride;

        for(Rpp32u i = 0; i < desc_ptr->h; i++)
        {
            for(Rpp32u j = 0; j < desc_ptr->w; j++)
            {
                *output_temp = *output_copy_temp_r;
                output_temp++;
                output_copy_temp_r++;
                *output_temp = *output_copy_temp_g;
                output_temp++;
                output_copy_temp_g++;
                *output_temp = *output_copy_temp_b;
                output_temp++;
                output_copy_temp_b++;
            }
        }
    }

    free(output_copy);
}

/// \brief Convert ROI type between XYWH and LTRB
inline void convert_roi(RpptROI* roi_tensor_ptr, RpptRoiType roi_type, int batch_size)
{
    if(roi_type == RpptRoiType::LTRB)
    {
        for(int i = 0; i < batch_size; i++)
        {
            RpptRoiXywh roi                = roi_tensor_ptr[i].xywhROI;
            roi_tensor_ptr[i].ltrbROI.lt.x = roi.xy.x;
            roi_tensor_ptr[i].ltrbROI.lt.y = roi.xy.y;
            roi_tensor_ptr[i].ltrbROI.rb.x = roi.roiWidth - roi.xy.x;
            roi_tensor_ptr[i].ltrbROI.rb.y = roi.roiHeight - roi.xy.y;
        }
    }
    else
    {
        for(int i = 0; i < batch_size; i++)
        {
            RpptRoiLtrb roi                     = roi_tensor_ptr[i].ltrbROI;
            roi_tensor_ptr[i].xywhROI.xy.x      = roi.lt.x;
            roi_tensor_ptr[i].xywhROI.xy.y      = roi.lt.y;
            roi_tensor_ptr[i].xywhROI.roiWidth  = roi.rb.x - roi.lt.x + 1;
            roi_tensor_ptr[i].xywhROI.roiHeight = roi.rb.y - roi.lt.y + 1;
        }
    }
}

/// \brief Validate pixel value to be in range [0, 255]
template<typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0))
                ? (static_cast<Rpp32f>(0))
                : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

/// \brief Convert U8 data to F16 format
inline void convert_u8_to_f16(
    Rpp8u* input_u8, void* input, size_t buffer_size, int offset_in_bytes, float conversion_factor)
{
    Rpp8u*  input_temp = input_u8 + offset_in_bytes;
    Rpp16f* input_f16_temp
        = reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(input) + offset_in_bytes);

    for(size_t i = 0; i < buffer_size; i++)
    {
        *input_f16_temp++
            = static_cast<Rpp16f>((static_cast<float>(*input_temp++)) * conversion_factor);
    }
}

/// \brief Convert U8 data to F32 format
inline void convert_u8_to_f32(
    Rpp8u* input_u8, void* input, size_t buffer_size, int offset_in_bytes, float conversion_factor)
{
    Rpp8u*  input_temp = input_u8 + offset_in_bytes;
    Rpp32f* input_f32_temp
        = reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(input) + offset_in_bytes);

    for(size_t i = 0; i < buffer_size; i++)
    {
        *input_f32_temp++ = (static_cast<Rpp32f>(*input_temp++)) * conversion_factor;
    }
}

/// \brief Convert U8 data to I8 format
inline void convert_u8_to_i8(Rpp8u* input_u8, void* input, size_t buffer_size, int offset_in_bytes)
{
    Rpp8u* input_temp    = input_u8 + offset_in_bytes;
    Rpp8s* input_i8_temp = static_cast<Rpp8s*>(input) + offset_in_bytes;

    for(size_t i = 0; i < buffer_size; i++)
    {
        *input_i8_temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*input_temp++)) - 128);
    }
}

/// \brief Convert F16 data to U8 format
inline void convert_f16_to_u8(void*  output,
                              Rpp8u* output_u8,
                              size_t buffer_size,
                              int    offset_in_bytes,
                              float  inv_conversion_factor)
{
    Rpp8u*  output_temp = output_u8 + offset_in_bytes;
    Rpp16f* output_f16_temp
        = reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(output) + offset_in_bytes);

    for(size_t i = 0; i < buffer_size; i++)
    {
        *output_temp = static_cast<Rpp8u>(
            validate_pixel_range(static_cast<float>(*output_f16_temp) * inv_conversion_factor));
        output_f16_temp++;
        output_temp++;
    }
}

/// \brief Convert F32 data to U8 format
inline void convert_f32_to_u8(void*  output,
                              Rpp8u* output_u8,
                              size_t buffer_size,
                              int    offset_in_bytes,
                              float  inv_conversion_factor)
{
    Rpp8u*  output_temp = output_u8 + offset_in_bytes;
    Rpp32f* output_f32_temp
        = reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(output) + offset_in_bytes);

    for(size_t i = 0; i < buffer_size; i++)
    {
        *output_temp
            = static_cast<Rpp8u>(validate_pixel_range(*output_f32_temp * inv_conversion_factor));
        output_f32_temp++;
        output_temp++;
    }
}

/// \brief Convert I8 data to U8 format
inline void
    convert_i8_to_u8(void* output, Rpp8u* output_u8, size_t buffer_size, int offset_in_bytes)
{
    Rpp8u* output_temp    = output_u8 + offset_in_bytes;
    Rpp8s* output_i8_temp = static_cast<Rpp8s*>(output) + offset_in_bytes;

    for(size_t i = 0; i < buffer_size; i++)
    {
        *output_temp = static_cast<Rpp8u>(
            validate_pixel_range((static_cast<Rpp32s>(*output_i8_temp)) + 128));
        output_i8_temp++;
        output_temp++;
    }
}

/// \brief Load image names from a directory
inline std::vector<std::string> load_image_names(const std::string& folder_path)
{
    std::vector<std::string> image_names;
    DIR*                     dir = opendir(folder_path.c_str());

    if(dir == nullptr)
    {
        std::cerr << "Error: Unable to open directory: " << folder_path << std::endl;
        return image_names;
    }

    struct dirent* entry;
    while((entry = readdir(dir)) != nullptr)
    {
        if(std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }
        image_names.push_back(entry->d_name);
    }

    closedir(dir);

    if(image_names.empty())
    {
        std::cerr << "Error: No images found in directory: " << folder_path << std::endl;
    }

    return image_names;
}

#endif // COMMON_RPP_UTILS_HPP
