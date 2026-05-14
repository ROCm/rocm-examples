// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, HIP kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out, hipTextureObject_t texCoarse)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = tex2D<float>(texCoarse, x, y) * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src,
                    int          width,
                    int          height,
                    int          stride,
                    int          newWidth,
                    int          newHeight,
                    int          newStride,
                    float        scale,
                    float       *out)
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

    hipTextureObject_t texCoarse;
    HIP_CHECK(hipCreateTextureObject(&texCoarse, &texRes, &texDescr, NULL));

    UpscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, scale, out, texCoarse);

    HIP_CHECK(hipDestroyTextureObject(texCoarse));
}
