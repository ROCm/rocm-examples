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

#ifndef MIVISIONX_UTILS_HPP
#define MIVISIONX_UTILS_HPP

#include <VX/vx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/// \brief Macro to check OpenVX status and exit on error
#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if(status_ != VX_SUCCESS)                                                               \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

/// \brief Macro to check OpenVX object status and exit on error
#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if(status_ != VX_SUCCESS)                                                               \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

/// \brief OpenVX log callback function
static void VX_CALLBACK log_callback(vx_context    context [[maybe_unused]],
                                     vx_reference  ref [[maybe_unused]],
                                     vx_status     status [[maybe_unused]],
                                     const vx_char string[])
{
    size_t len = strlen(string);
    if(len > 0)
    {
        printf("%s", string);
        if(string[len - 1] != '\n')
        {
            printf("\n");
        }
        fflush(stdout);
    }
}

/// \brief Helper function to initialize vx_rectangle_t
inline void init_vx_rectangle(
    vx_rectangle_t& rect, vx_uint32 start_x, vx_uint32 start_y, vx_uint32 end_x, vx_uint32 end_y)
{
    rect.start_x = start_x;
    rect.start_y = start_y;
    rect.end_x   = end_x;
    rect.end_y   = end_y;
}

/// \brief Helper function to initialize vx_imagepatch_addressing_t for grayscale images
inline void init_vx_image_layout_gray(vx_imagepatch_addressing_t& layout, vx_uint32 stride_y)
{
    layout.stride_x = 1;
    layout.stride_y = stride_y;
}

/// \brief Helper function to initialize vx_imagepatch_addressing_t for RGB images
inline void init_vx_image_layout_rgb(vx_imagepatch_addressing_t& layout, vx_uint32 stride_y)
{
    layout.stride_x = 3;
    layout.stride_y = stride_y;
}

#endif // MIVISIONX_UTILS_HPP
