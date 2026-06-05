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

#include <hip/hip_runtime.h>

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// HIP kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void DownscaleKernel(int width, int height, int stride, float *out, hipTextureObject_t texFine)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) {
        return;
    }

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    out[ix + iy * stride] = 0.25f
                          * (tex2D<float>(texFine, x - dx * 0.25f, y) + tex2D<float>(texFine, x + dx * 0.25f, y)
                             + tex2D<float>(texFine, x, y - dy * 0.25f) + tex2D<float>(texFine, x, y + dy * 0.25f));
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static void
Downscale(const float *src, int width, int height, int stride, int newWidth, int newHeight, int newStride, float *out)
{
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

    hipResourceDesc texRes = {};
    texRes.resType                  = hipResourceTypePitch2D;
    texRes.res.pitch2D.devPtr       = (void *)src;
    texRes.res.pitch2D.desc         = hipCreateChannelDesc<float>();
    texRes.res.pitch2D.width        = width;
    texRes.res.pitch2D.height       = height;
    texRes.res.pitch2D.pitchInBytes = stride * sizeof(float);

    hipTextureDesc texDescr = {};
    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0]   = hipAddressModeMirror;
    texDescr.addressMode[1]   = hipAddressModeMirror;
    texDescr.readMode         = hipReadModeElementType;

    hipTextureObject_t texFine;
    HIP_CHECK(hipCreateTextureObject(&texFine, &texRes, &texDescr, NULL));

    DownscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, out, texFine);

    HIP_CHECK(hipDestroyTextureObject(texFine));
}
