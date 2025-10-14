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

#include "mivisionx_utils.hpp"

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <vx_ext_opencv.h>

#include "CmdParser/cmdparser.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("image",
                                     "image",
                                     "",
                                     "Path to input image file (if not provided, uses default)");
    parser.set_optional<bool>("live", "live", false, "Use live camera feed instead of image file");
    parser.set_optional<int>("width", "width", 480, "Image width");
    parser.set_optional<int>("height", "height", 360, "Image height");
    parser.run_and_exit_if_error();

    // Get command line arguments
    const std::string image_path_arg = parser.get<std::string>("image");
    const bool        use_live       = parser.get<bool>("live");
    const int         width          = parser.get<int>("width");
    const int         height         = parser.get<int>("height");

    // Determine image path
    std::string image_path;
    if(!image_path_arg.empty())
    {
        image_path = image_path_arg;
    }
    else if(!use_live)
    {
        image_path = std::string(EXAMPLE_DATA_DIR) + "/face.jpg";
    }

    // Create OpenVX context
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    // Create OpenVX graph
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // Create images
    vx_image input_rgb_image       = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image output_filtered_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    vx_image yuv_image             = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image            = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_filtered_image);
    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);

    // Create threshold
    vx_threshold hyst  = vxCreateThresholdForImage(context,
                                                  VX_THRESHOLD_TYPE_RANGE,
                                                  VX_DF_IMAGE_U8,
                                                  VX_DF_IMAGE_U8);
    vx_int32     lower = 80;
    vx_int32     upper = 100;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
    ERROR_CHECK_OBJECT(hyst);

    vx_int32 gradient_size = 3;

    // Create graph nodes
    vx_node nodes[] = {vxColorConvertNode(graph, input_rgb_image, yuv_image),
                       vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
                       vxCannyEdgeDetectorNode(graph,
                                               luma_image,
                                               hyst,
                                               gradient_size,
                                               VX_NORM_L1,
                                               output_filtered_image)};

    for(vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    Mat input;

    if(use_live)
    {
        // Live camera mode
        VideoCapture cap(0);
        if(!cap.isOpened())
        {
            printf("Unable to open camera\n");
            return 1;
        }

        for(;;)
        {
            cap >> input;
            resize(input, input, Size(width, height));
            imshow("inputWindow", input);
            if(waitKey(30) >= 0)
            {
                break;
            }

            vx_rectangle_t             cv_rgb_image_region;
            vx_imagepatch_addressing_t cv_rgb_image_layout;
            init_vx_rectangle(cv_rgb_image_region, 0, 0, width, height);
            init_vx_image_layout_rgb(cv_rgb_image_layout, input.step);
            vx_uint8* cv_rgb_image_buffer = input.data;

            ERROR_CHECK_STATUS(vxCopyImagePatch(input_rgb_image,
                                                &cv_rgb_image_region,
                                                0,
                                                &cv_rgb_image_layout,
                                                cv_rgb_image_buffer,
                                                VX_WRITE_ONLY,
                                                VX_MEMORY_TYPE_HOST));

            ERROR_CHECK_STATUS(vxProcessGraph(graph));

            vx_rectangle_t             rect = {0, 0, (vx_uint32)width, (vx_uint32)height};
            vx_map_id                  map_id;
            vx_imagepatch_addressing_t addr;
            void*                      ptr;
            ERROR_CHECK_STATUS(vxMapImagePatch(output_filtered_image,
                                               &rect,
                                               0,
                                               &map_id,
                                               &addr,
                                               &ptr,
                                               VX_READ_ONLY,
                                               VX_MEMORY_TYPE_HOST,
                                               VX_NOGAP_X));

            Mat mat(height, width, CV_8U, ptr, addr.stride_y);
            imshow("CannyDetect", mat);
            if(waitKey(30) >= 0)
            {
                break;
            }
            ERROR_CHECK_STATUS(vxUnmapImagePatch(output_filtered_image, map_id));
        }
    }
    else
    {
        // Image mode
        input = imread(image_path);
        if(input.empty())
        {
            printf("Image not found: %s\n", image_path.c_str());
            return 1;
        }

        resize(input, input, Size(width, height));
        imshow("inputWindow", input);

        vx_rectangle_t             cv_rgb_image_region;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        init_vx_rectangle(cv_rgb_image_region, 0, 0, width, height);
        init_vx_image_layout_rgb(cv_rgb_image_layout, input.step);
        vx_uint8* cv_rgb_image_buffer = input.data;

        ERROR_CHECK_STATUS(vxCopyImagePatch(input_rgb_image,
                                            &cv_rgb_image_region,
                                            0,
                                            &cv_rgb_image_layout,
                                            cv_rgb_image_buffer,
                                            VX_WRITE_ONLY,
                                            VX_MEMORY_TYPE_HOST));

        ERROR_CHECK_STATUS(vxProcessGraph(graph));

        vx_rectangle_t             rect = {0, 0, (vx_uint32)width, (vx_uint32)height};
        vx_map_id                  map_id;
        vx_imagepatch_addressing_t addr;
        void*                      ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(output_filtered_image,
                                           &rect,
                                           0,
                                           &map_id,
                                           &addr,
                                           &ptr,
                                           VX_READ_ONLY,
                                           VX_MEMORY_TYPE_HOST,
                                           VX_NOGAP_X));

        Mat mat(height, width, CV_8U, ptr, addr.stride_y);
        imshow("CannyDetect", mat);
        waitKey(0);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(output_filtered_image, map_id));
    }

    // Cleanup
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_filtered_image));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&hyst));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}
