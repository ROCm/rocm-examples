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
    parser.set_optional<int>("width", "width", 1280, "Image width");
    parser.set_optional<int>("height", "height", 720, "Image height");
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

    // Load OpenCV extension kernels
    vxLoadKernels(context, "vx_opencv");

    // Create OpenVX graph
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // Create images
    vx_image inter_luma = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(inter_luma);

    // Create keypoints array
    vx_array keypoints = vxCreateArray(context, VX_TYPE_KEYPOINT, 10000);
    ERROR_CHECK_OBJECT(keypoints);

    // ORB parameters
    vx_int32   n_features     = 1000;
    vx_float32 scale_factor   = 1.2f;
    vx_int32   n_levels       = 2;
    vx_int32   edge_threshold = 31;
    vx_int32   first_level    = 0;
    vx_int32   wta_k          = 2;
    vx_int32   score_type     = 0;
    vx_int32   patch_size     = 31;

    // Create graph nodes
    vx_node nodes[] = {vxExtCvNode_orbDetect(graph,
                                             inter_luma,
                                             inter_luma,
                                             keypoints,
                                             n_features,
                                             scale_factor,
                                             n_levels,
                                             edge_threshold,
                                             first_level,
                                             wta_k,
                                             score_type,
                                             patch_size)};

    for(vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    Mat input;
    Mat output;

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
            cap >> output;

            if(input.empty())
            {
                printf("Failed to capture frame\n");
                break;
            }

            cvtColor(input, input, COLOR_RGB2GRAY);
            cv::resize(input, input, Size(width, height));
            cv::resize(output, output, Size(width, height));

            vx_rectangle_t             cv_image_region;
            vx_imagepatch_addressing_t cv_image_layout;
            init_vx_rectangle(cv_image_region, 0, 0, width, height);
            init_vx_image_layout_gray(cv_image_layout, input.step);
            vx_uint8* cv_image_buffer = input.data;

            ERROR_CHECK_STATUS(vxCopyImagePatch(inter_luma,
                                                &cv_image_region,
                                                0,
                                                &cv_image_layout,
                                                cv_image_buffer,
                                                VX_WRITE_ONLY,
                                                VX_MEMORY_TYPE_HOST));

            ERROR_CHECK_STATUS(vxProcessGraph(graph));

            vx_size num_corners = 0;
            ERROR_CHECK_STATUS(
                vxQueryArray(keypoints, VX_ARRAY_NUMITEMS, &num_corners, sizeof(num_corners)));

            if(num_corners > 0)
            {
                vx_size   kp_stride;
                vx_map_id kp_map;
                vx_uint8* kp_buf;
                ERROR_CHECK_STATUS(vxMapArrayRange(keypoints,
                                                   0,
                                                   num_corners,
                                                   &kp_map,
                                                   &kp_stride,
                                                   (void**)&kp_buf,
                                                   VX_READ_ONLY,
                                                   VX_MEMORY_TYPE_HOST,
                                                   0));

                for(vx_size i = 0; i < num_corners; i++)
                {
                    vx_keypoint_t* kp = (vx_keypoint_t*)(kp_buf + i * kp_stride);
                    cv::Point      center(kp->x, kp->y);
                    cv::circle(output, center, 1, cv::Scalar(0, 255, 0), 2);
                }

                ERROR_CHECK_STATUS(vxUnmapArrayRange(keypoints, kp_map));
            }

            imshow("OrbDetect", output);
            if(waitKey(30) >= 0)
            {
                break;
            }
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

        output = input.clone();
        cvtColor(input, input, COLOR_RGB2GRAY);
        cv::resize(input, input, Size(width, height));
        cv::resize(output, output, Size(width, height));

        vx_rectangle_t             cv_image_region;
        vx_imagepatch_addressing_t cv_image_layout;
        init_vx_rectangle(cv_image_region, 0, 0, width, height);
        init_vx_image_layout_gray(cv_image_layout, input.step);
        vx_uint8* cv_image_buffer = input.data;

        ERROR_CHECK_STATUS(vxCopyImagePatch(inter_luma,
                                            &cv_image_region,
                                            0,
                                            &cv_image_layout,
                                            cv_image_buffer,
                                            VX_WRITE_ONLY,
                                            VX_MEMORY_TYPE_HOST));

        ERROR_CHECK_STATUS(vxProcessGraph(graph));

        vx_size num_corners = 0;
        ERROR_CHECK_STATUS(
            vxQueryArray(keypoints, VX_ARRAY_NUMITEMS, &num_corners, sizeof(num_corners)));

        if(num_corners > 0)
        {
            vx_size   kp_stride;
            vx_map_id kp_map;
            vx_uint8* kp_buf;
            ERROR_CHECK_STATUS(vxMapArrayRange(keypoints,
                                               0,
                                               num_corners,
                                               &kp_map,
                                               &kp_stride,
                                               (void**)&kp_buf,
                                               VX_READ_ONLY,
                                               VX_MEMORY_TYPE_HOST,
                                               0));

            for(vx_size i = 0; i < num_corners; i++)
            {
                vx_keypoint_t* kp = (vx_keypoint_t*)(kp_buf + i * kp_stride);
                cv::Point      center(kp->x, kp->y);
                cv::circle(output, center, 1, cv::Scalar(0, 255, 0), 2);
            }

            ERROR_CHECK_STATUS(vxUnmapArrayRange(keypoints, kp_map));
        }

        imshow("OrbDetect", output);
        waitKey(0);
    }

    // Cleanup
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseArray(&keypoints));
    ERROR_CHECK_STATUS(vxReleaseImage(&inter_luma));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}
