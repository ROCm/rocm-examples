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
/// \brief warp image with a given displacement field, HIP kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void WarpingKernel(int                 width,
                              int                 height,
                              int                 stride,
                              const float        *u,
                              const float        *v,
                              float              *out,
                              hipTextureObject_t texToWarp)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + u[pos] + 0.5f) / (float)width;
    float y = ((float)iy + v[pos] + 0.5f) / (float)height;

    out[pos] = tex2D<float>(texToWarp, x, y);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, HIP kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, int w, int h, int s, const float *u, const float *v, float *out)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    hipResourceDesc texRes = {};
    texRes.resType                  = hipResourceTypePitch2D;
    texRes.res.pitch2D.devPtr       = (void *)src;
    texRes.res.pitch2D.desc         = hipCreateChannelDesc<float>();
    texRes.res.pitch2D.width        = w;
    texRes.res.pitch2D.height       = h;
    texRes.res.pitch2D.pitchInBytes = s * sizeof(float);

    hipTextureDesc texDescr = {};
    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0]   = hipAddressModeMirror;
    texDescr.addressMode[1]   = hipAddressModeMirror;
    texDescr.readMode         = hipReadModeElementType;

    hipTextureObject_t texToWarp;
    HIP_CHECK(hipCreateTextureObject(&texToWarp, &texRes, &texDescr, NULL));

    WarpingKernel<<<blocks, threads>>>(w, h, s, u, v, out, texToWarp);

    HIP_CHECK(hipDestroyTextureObject(texToWarp));
}
