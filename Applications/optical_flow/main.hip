// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"
#include <hip/hip_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "common.h"
#include "flowGold.h"
#include "flowHIP.h"

#include <string>
#ifndef _WIN32
#    include <unistd.h>
#endif

constexpr const char *SAMPLE_NAME = "HSOpticalFlow";

// Resolve a data file path relative to this source file's directory.
// Falls back to /proc/self/exe when __FILE__ is a bare basename (non-absolute
// compiler invocation), which happens when hipcc is called with a relative path.
static std::string data_path(const char *filename)
{
    std::string src = __FILE__;
#ifdef _WIN32
    char sep = src.find_last_of('/') == std::string::npos ? '\\' : '/';
    if(src.find("optical_flow") == std::string::npos)
    {
        char buffer[MAX_PATH];
        GetModuleFileNameA(NULL, buffer, MAX_PATH);
        src = std::string(buffer);
    }
#else
    constexpr char sep = '/';
    if(src.find(sep) == std::string::npos)
    {
        char buf[4096] = {};
        if(readlink("/proc/self/exe", buf, sizeof(buf) - 1) > 0)
            src = std::string(buf);
    }
#endif
    return src.substr(0, src.find_last_of(sep)) + sep + "data" + sep + filename;
}

// CPU-GPU discrepancy threshold for self-test
constexpr float THRESHOLD = 0.05f;

///////////////////////////////////////////////////////////////////////////////
/// \brief save optical flow in format described on vision.middlebury.edu/flow
/// \param[in] name output file name
/// \param[in] w    optical flow field width
/// \param[in] h    optical flow field height
/// \param[in] s    optical flow field row stride
/// \param[in] u    horizontal displacement
/// \param[in] v    vertical displacement
///////////////////////////////////////////////////////////////////////////////
void WriteFloFile(const char *name, int w, int h, int s, const float *u, const float *v)
{
    FILE *stream;
    stream = fopen(name, "wb");

    if (stream == 0) {
        printf("Could not save flow to \"%s\"\n", name);
        return;
    }

    float data = 202021.25f;
    bool  ok   = (fwrite(&data, sizeof(float), 1, stream) == 1)
              && (fwrite(&w, sizeof(w), 1, stream) == 1)
              && (fwrite(&h, sizeof(h), 1, stream) == 1);

    for (int i = 0; i < h && ok; ++i) {
        for (int j = 0; j < w && ok; ++j) {
            const int pos = j + i * s;
            ok            = (fwrite(u + pos, sizeof(float), 1, stream) == 1)
               && (fwrite(v + pos, sizeof(float), 1, stream) == 1);
        }
    }

    if (!ok)
        printf("Write error saving flow to \"%s\"\n", name);

    fclose(stream);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief
/// load 4-channel unsigned byte image
/// and convert it to single channel FP32 image
/// \param[out] img_data pointer to raw image data
/// \param[out] img_w    image width
/// \param[out] img_h    image height
/// \param[out] img_s    image row stride
/// \param[in]  name     image file name
/// \return true if image is successfully loaded or false otherwise
///////////////////////////////////////////////////////////////////////////////
bool LoadImageAsFP32(float *&img_data, int &img_w, int &img_h, int &img_s, const char *name)
{
    printf("Loading \"%s\" ...\n", name);
    int            channels;
    unsigned char *data = stbi_load(name, &img_w, &img_h, &channels, 3); // force RGB
    if (!data) {
        printf("Failed to load image\n");
        return false;
    }

    img_s    = iAlignUp(img_w);
    img_data = new float[img_s * img_h];

    for (int i = 0; i < img_h; ++i)
        for (int j = 0; j < img_w; ++j)
            img_data[j + i * img_s] = data[(j + i * img_w) * 3] / 255.0f; // R channel

    stbi_image_free(data);
    return true;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compare given flow field with gold (L1 norm)
/// \param[in] width    optical flow field width
/// \param[in] height   optical flow field height
/// \param[in] stride   optical flow field row stride
/// \param[in] h_uGold  horizontal displacement, gold
/// \param[in] h_vGold  vertical displacement, gold
/// \param[in] h_u      horizontal displacement
/// \param[in] h_v      vertical displacement
/// \return true if discrepancy is lower than a given threshold
///////////////////////////////////////////////////////////////////////////////
bool CompareWithGold(int          width,
                     int          height,
                     int          stride,
                     const float *h_uGold,
                     const float *h_vGold,
                     const float *h_u,
                     const float *h_v)
{
    float error = 0.0f;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const int pos = j + i * stride;
            error += fabsf(h_u[pos] - h_uGold[pos]) + fabsf(h_v[pos] - h_vGold[pos]);
        }
    }

    error /= (float)(width * height);

    printf("L1 error : %.6f\n", error);

    return (error < THRESHOLD);
}

///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main()
{
    // welcome message
    printf("%s Starting...\n\n", SAMPLE_NAME);

    // see which GPU is being used
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Using device: %s\n", props.name);

    // image dimensions
    int width;
    int height;
    // row access stride
    int stride;

    // flow is computed from source image to target image
    float *h_source; // source image, host memory
    float *h_target; // target image, host memory

    // load image from file
    if (!LoadImageAsFP32(h_source, width, height, stride, data_path("frame10.ppm").c_str())) {
        exit(EXIT_FAILURE);
    }

    const int source_width  = width;
    const int source_height = height;

    if (!LoadImageAsFP32(h_target, width, height, stride, data_path("frame11.ppm").c_str())) {
        delete[] h_source;
        exit(EXIT_FAILURE);
    }

    if (width != source_width || height != source_height) {
        printf("Frame dimension mismatch: frame10 is %dx%d but frame11 is %dx%d\n",
               source_width,
               source_height,
               width,
               height);
        delete[] h_source;
        delete[] h_target;
        exit(EXIT_FAILURE);
    }

    // allocate host memory for CPU results
    float *h_uGold = new float[stride * height];
    float *h_vGold = new float[stride * height];

    // allocate host memory for GPU results
    float *h_u = new float[stride * height];
    float *h_v = new float[stride * height];

    // smoothness
    // if image brightness is not within [0,1]
    // this paramter should be scaled appropriately
    const float alpha = 0.2f;

    // number of pyramid levels
    const int nLevels = 5;

    // number of solver iterations on each level
    const int nSolverIters = 500;

    // number of warping iterations
    const int nWarpIters = 3;

    ComputeFlowGold(
        h_source, h_target, width, height, stride, alpha, nLevels, nWarpIters, nSolverIters, h_uGold, h_vGold);

    ComputeFlowHIP(h_source, h_target, width, height, stride, alpha, nLevels, nWarpIters, nSolverIters, h_u, h_v);

    // compare results (L1 norm)
    bool status = CompareWithGold(width, height, stride, h_uGold, h_vGold, h_u, h_v);

    WriteFloFile("FlowGPU.flo", width, height, stride, h_u, h_v);

    WriteFloFile("FlowCPU.flo", width, height, stride, h_uGold, h_vGold);

    // free resources
    delete[] h_uGold;
    delete[] h_vGold;

    delete[] h_u;
    delete[] h_v;

    delete[] h_source;
    delete[] h_target;

    // report self-test status
    exit(status ? EXIT_SUCCESS : EXIT_FAILURE);
}
