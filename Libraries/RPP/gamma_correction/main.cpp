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

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "rpp_utils.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("input", "input", EXAMPLE_DATA_DIR, "Input image folder path");
    parser.set_optional<std::string>("output", "output", "./output", "Output folder path");
    parser.set_optional<int>(
        "bit-depth",
        "bit-depth",
        0,
        "Input bit depth (0=U8, 1=F16, 2=F32, 3=U8->F16, 4=U8->F32, 5=I8, 6=U8->I8)");
    parser.set_optional<int>("output-format",
                             "output-format",
                             0,
                             "Output format toggle (0=same as input, 1=convert)");
    parser.set_optional<int>("layout", "layout", 0, "Layout type (0=PKD3, 1=PLN3, 2=PLN1)");
    parser.set_optional<float>("gamma",
                               "gamma",
                               1.5f,
                               "Gamma correction value (typically 0.5 to 2.5)");
    parser.run_and_exit_if_error();

    // Get parsed arguments
    const std::string input_folder         = parser.get<std::string>("input");
    const std::string output_folder        = parser.get<std::string>("output");
    const int         input_bit_depth      = parser.get<int>("bit-depth");
    const int         output_format_toggle = parser.get<int>("output-format");
    const int         layout_type          = parser.get<int>("layout");
    const float       gamma_value          = parser.get<float>("gamma");

    // Validate gamma value
    if(gamma_value <= 0.0f)
    {
        std::cerr << "Error: Gamma value must be positive" << std::endl;
        return error_exit_code;
    }

    // Validate layout and output format combination
    if(layout_type == 2 && output_format_toggle != 0)
    {
        std::cerr << "Error: PLN1 layout does not support output format toggle. "
                  << "Please set output-format to 0." << std::endl;
        return error_exit_code;
    }

    // Determine the number of input channels based on layout type
    const int input_channels = set_input_channels(layout_type);

    // Initialize tensor descriptors
    RpptDesc    src_desc, dst_desc;
    RpptDescPtr src_desc_ptr = &src_desc;
    RpptDescPtr dst_desc_ptr = &dst_desc;

    // Set src/dst layout types in tensor descriptors
    set_descriptor_layout(src_desc_ptr, dst_desc_ptr, layout_type, output_format_toggle);

    // Set src/dst data types in tensor descriptors
    set_descriptor_data_type(input_bit_depth, src_desc_ptr, dst_desc_ptr);

    // Load image names from input folder
    std::vector<std::string> image_names = load_image_names(input_folder);
    if(image_names.empty())
    {
        std::cerr << "Error: No images found in input folder: " << input_folder << std::endl;
        return error_exit_code;
    }

    const int num_images = static_cast<int>(image_names.size());
    std::cout << "Found " << num_images << " images in input folder" << std::endl;

    // Build full image paths
    std::vector<std::string> image_paths;
    for(const auto& name : image_names)
    {
        image_paths.push_back(input_folder + "/" + name);
    }

    // Initialize ROI tensors for src/dst
    RpptROI* roi_tensor_ptr_src;
    RpptROI* roi_tensor_ptr_dst;
    HIP_CHECK(hipHostMalloc(&roi_tensor_ptr_src, num_images * sizeof(RpptROI)));
    HIP_CHECK(hipHostMalloc(&roi_tensor_ptr_dst, num_images * sizeof(RpptROI)));

    // Initialize the ImagePatch for dst
    RpptImagePatch* dst_img_sizes;
    HIP_CHECK(hipHostMalloc(&dst_img_sizes, num_images * sizeof(RpptImagePatch)));

    // Set ROI tensor types for src/dst
    RpptRoiType roi_type_src = RpptRoiType::XYWH;

    // Read images to determine max dimensions and set ROIs
    int max_height     = 0;
    int max_width      = 0;
    int max_dst_height = 0;
    int max_dst_width  = 0;

    for(int i = 0; i < num_images; i++)
    {
        cv::Mat image;
        if(layout_type == 0 || layout_type == 1)
        {
            image = cv::imread(image_paths[i], 1);
        }
        else
        {
            image = cv::imread(image_paths[i], 0);
        }

        if(image.empty())
        {
            std::cerr << "Error: Unable to read image: " << image_paths[i] << std::endl;
            return error_exit_code;
        }

        roi_tensor_ptr_src[i].xywhROI.xy.x      = 0;
        roi_tensor_ptr_src[i].xywhROI.xy.y      = 0;
        roi_tensor_ptr_src[i].xywhROI.roiWidth  = image.cols;
        roi_tensor_ptr_src[i].xywhROI.roiHeight = image.rows;
        roi_tensor_ptr_dst[i].xywhROI.xy.x      = 0;
        roi_tensor_ptr_dst[i].xywhROI.xy.y      = 0;
        roi_tensor_ptr_dst[i].xywhROI.roiWidth  = image.cols;
        roi_tensor_ptr_dst[i].xywhROI.roiHeight = image.rows;
        dst_img_sizes[i].width                  = roi_tensor_ptr_dst[i].xywhROI.roiWidth;
        dst_img_sizes[i].height                 = roi_tensor_ptr_dst[i].xywhROI.roiHeight;

        max_height     = std::max(max_height, roi_tensor_ptr_src[i].xywhROI.roiHeight);
        max_width      = std::max(max_width, roi_tensor_ptr_src[i].xywhROI.roiWidth);
        max_dst_height = std::max(max_dst_height, roi_tensor_ptr_dst[i].xywhROI.roiHeight);
        max_dst_width  = std::max(max_dst_width, roi_tensor_ptr_dst[i].xywhROI.roiWidth);
    }

    // Check if dimensions are valid
    if(max_height <= 0 || max_width <= 0)
    {
        std::cerr << "Error: Invalid image dimensions detected" << std::endl;
        return error_exit_code;
    }

    const Rpp32u output_channels = input_channels;
    const Rpp32u offset_in_bytes = 0;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(src_desc_ptr,
                                    num_images,
                                    max_height,
                                    max_width,
                                    input_channels,
                                    offset_in_bytes);
    set_descriptor_dims_and_strides(dst_desc_ptr,
                                    num_images,
                                    max_dst_height,
                                    max_dst_width,
                                    output_channels,
                                    offset_in_bytes);

    // Set buffer sizes in pixels for src/dst
    const Rpp64u io_buffer_size
        = static_cast<Rpp64u>(src_desc_ptr->h) * static_cast<Rpp64u>(src_desc_ptr->w)
          * static_cast<Rpp64u>(src_desc_ptr->c) * static_cast<Rpp64u>(num_images);
    const Rpp64u o_buffer_size
        = static_cast<Rpp64u>(dst_desc_ptr->h) * static_cast<Rpp64u>(dst_desc_ptr->w)
          * static_cast<Rpp64u>(dst_desc_ptr->c) * static_cast<Rpp64u>(num_images);

    // Set buffer sizes in bytes for src/dst (including offsets)
    const Rpp64u io_buffer_size_in_bytes_u8 = io_buffer_size + src_desc_ptr->offsetInBytes;
    const Rpp64u o_buffer_size_in_bytes_u8  = o_buffer_size + dst_desc_ptr->offsetInBytes;
    const Rpp64u input_buffer_size = io_buffer_size * get_size_of_data_type(src_desc_ptr->dataType)
                                     + src_desc_ptr->offsetInBytes;
    const Rpp64u output_buffer_size = o_buffer_size * get_size_of_data_type(dst_desc_ptr->dataType)
                                      + dst_desc_ptr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u* input_u8  = static_cast<Rpp8u*>(calloc(io_buffer_size_in_bytes_u8, 1));
    Rpp8u* output_u8 = static_cast<Rpp8u*>(calloc(o_buffer_size_in_bytes_u8, 1));

    // Read images using OpenCV
    read_image_batch_opencv(input_u8, src_desc_ptr, image_paths);
    std::cout << "Decoded images using OpenCV" << std::endl;

    // If the input layout requested is PLN3, convert PKD3 inputs to PLN3
    if(layout_type == 1)
    {
        convert_pkd3_to_pln3(input_u8, src_desc_ptr);
    }

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert back to 0-255 range
    const Rpp32f conversion_factor     = 1.0f / 255.0f;
    const Rpp32f inv_conversion_factor = 1.0f / conversion_factor;

    void* input  = static_cast<Rpp8u*>(calloc(input_buffer_size, 1));
    void* output = static_cast<Rpp8u*>(calloc(output_buffer_size, 1));

    // Convert inputs to corresponding bit depth specified by user
    if(input_bit_depth == 0)
    {
        std::memcpy(input, input_u8, input_buffer_size);
    }
    else if(input_bit_depth == 1)
    {
        convert_u8_to_f16(input_u8,
                          input,
                          io_buffer_size,
                          src_desc_ptr->offsetInBytes,
                          conversion_factor);
    }
    else if(input_bit_depth == 2)
    {
        convert_u8_to_f32(input_u8,
                          input,
                          io_buffer_size,
                          src_desc_ptr->offsetInBytes,
                          conversion_factor);
    }
    else if(input_bit_depth == 5)
    {
        convert_u8_to_i8(input_u8, input, io_buffer_size, src_desc_ptr->offsetInBytes);
    }

    // Allocate HIP memory for src/dst and copy decoded inputs to HIP buffers
    void* d_input;
    void* d_output;
    HIP_CHECK(hipMalloc(&d_input, input_buffer_size));
    HIP_CHECK(hipMalloc(&d_output, output_buffer_size));
    HIP_CHECK(hipMemcpy(d_input, input, input_buffer_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output, output, output_buffer_size, hipMemcpyHostToDevice));

    // Create RPP handle
    rppHandle_t handle;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    RPP_CHECK(rppCreate(&handle, num_images, 0, stream, backend));

    // Set parameters for gamma correction operation
    Rpp32f* gamma_tensor;
    HIP_CHECK(hipHostMalloc(&gamma_tensor, num_images * sizeof(Rpp32f)));

    for(int i = 0; i < num_images; i++)
    {
        gamma_tensor[i] = gamma_value;
    }

    // Execute gamma correction kernel
    if(input_bit_depth == 0 || input_bit_depth == 1 || input_bit_depth == 2 || input_bit_depth == 5)
    {
        RPP_CHECK(rppt_gamma_correction(d_input,
                                        src_desc_ptr,
                                        d_output,
                                        dst_desc_ptr,
                                        gamma_tensor,
                                        roi_tensor_ptr_src,
                                        roi_type_src,
                                        handle,
                                        backend));
        std::cout << "Executed gamma correction kernel on HIP backend" << std::endl;
    }
    else
    {
        std::cerr << "Error: Unsupported bit depth configuration" << std::endl;
        return error_exit_code;
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Reconvert other bit depths to 8u for output display purposes
    if(input_bit_depth == 0)
    {
        HIP_CHECK(hipMemcpy(output, d_output, output_buffer_size, hipMemcpyDeviceToHost));
        std::memcpy(output_u8, output, output_buffer_size);
    }
    else if(input_bit_depth == 1 || input_bit_depth == 3)
    {
        HIP_CHECK(hipMemcpy(output, d_output, output_buffer_size, hipMemcpyDeviceToHost));
        convert_f16_to_u8(output,
                          output_u8,
                          o_buffer_size,
                          dst_desc_ptr->offsetInBytes,
                          inv_conversion_factor);
    }
    else if(input_bit_depth == 2 || input_bit_depth == 4)
    {
        HIP_CHECK(hipMemcpy(output, d_output, output_buffer_size, hipMemcpyDeviceToHost));
        convert_f32_to_u8(output,
                          output_u8,
                          o_buffer_size,
                          dst_desc_ptr->offsetInBytes,
                          inv_conversion_factor);
    }
    else if(input_bit_depth == 5 || input_bit_depth == 6)
    {
        HIP_CHECK(hipMemcpy(output, d_output, output_buffer_size, hipMemcpyDeviceToHost));
        convert_i8_to_u8(output, output_u8, o_buffer_size, dst_desc_ptr->offsetInBytes);
    }

    // Calculate exact dstROI in XYWH format for OpenCV dump
    if(roi_type_src == RpptRoiType::LTRB)
    {
        convert_roi(roi_tensor_ptr_dst, RpptRoiType::XYWH, dst_desc_ptr->n);
    }

    // Check if the ROI values for each input is within the bounds of the max buffer allocated
    RpptROI    roi_default;
    RpptROIPtr roi_ptr_default         = &roi_default;
    roi_ptr_default->xywhROI.xy.x      = 0;
    roi_ptr_default->xywhROI.xy.y      = 0;
    roi_ptr_default->xywhROI.roiWidth  = static_cast<Rpp32s>(dst_desc_ptr->w);
    roi_ptr_default->xywhROI.roiHeight = static_cast<Rpp32s>(dst_desc_ptr->h);

    for(Rpp32u i = 0; i < dst_desc_ptr->n; i++)
    {
        roi_tensor_ptr_dst[i].xywhROI.roiWidth
            = std::min(roi_ptr_default->xywhROI.roiWidth - roi_tensor_ptr_dst[i].xywhROI.xy.x,
                       roi_tensor_ptr_dst[i].xywhROI.roiWidth);
        roi_tensor_ptr_dst[i].xywhROI.roiHeight
            = std::min(roi_ptr_default->xywhROI.roiHeight - roi_tensor_ptr_dst[i].xywhROI.xy.y,
                       roi_tensor_ptr_dst[i].xywhROI.roiHeight);
        roi_tensor_ptr_dst[i].xywhROI.xy.x
            = std::max(roi_ptr_default->xywhROI.xy.x, roi_tensor_ptr_dst[i].xywhROI.xy.x);
        roi_tensor_ptr_dst[i].xywhROI.xy.y
            = std::max(roi_ptr_default->xywhROI.xy.y, roi_tensor_ptr_dst[i].xywhROI.xy.y);
    }

    // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
    if(layout_type == 0 || layout_type == 1)
    {
        if((dst_desc_ptr->c == 3) && (dst_desc_ptr->layout == RpptLayout::NCHW))
        {
            convert_pln3_to_pkd3(output_u8, dst_desc_ptr);
        }
    }

    // Write output images
    write_image_batch_opencv(output_folder, output_u8, dst_desc_ptr, image_names, dst_img_sizes);
    std::cout << "Output images written to: " << output_folder << std::endl;

    // Cleanup
    RPP_CHECK(rppDestroy(handle, backend));
    HIP_CHECK(hipHostFree(roi_tensor_ptr_src));
    HIP_CHECK(hipHostFree(roi_tensor_ptr_dst));
    HIP_CHECK(hipHostFree(dst_img_sizes));
    HIP_CHECK(hipHostFree(gamma_tensor));
    free(input);
    free(output);
    free(input_u8);
    free(output_u8);
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout << "Gamma correction example completed successfully" << std::endl;

    return 0;
}
