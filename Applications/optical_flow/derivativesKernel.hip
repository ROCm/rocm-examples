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
/// \brief compute image derivatives (software sampler path)
///
/// Used on devices that lack hardware texture units (e.g. CDNA / MI300X).
/// \param[in]  width      image width
/// \param[in]  height     image height
/// \param[in]  stride     image stride in elements
/// \param[out] Ix         x derivative
/// \param[out] Iy         y derivative
/// \param[out] Iz         temporal derivative
/// \param[in]  texSource  source image device pointer (SW path)
/// \param[in]  srcWidth   source image width (SW path)
/// \param[in]  srcHeight  source image height (SW path)
/// \param[in]  srcStride  source image stride in elements (SW path)
/// \param[in]  texTarget  target image device pointer (SW path)
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernelSW(int          width,
                                            int          height,
                                            int          stride,
                                            float       *Ix,
                                            float       *Iy,
                                            float       *Iz,
                                            const float *texSource,
                                            int          srcWidth,
                                            int          srcHeight,
                                            int          srcStride,
                                            const float *texTarget)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

#define SAMPLE(tex, u, v) tex2D_bilinear(tex, srcWidth, srcHeight, srcStride, u, v)

    float t0, t1;
    // x derivative
    t0 = SAMPLE(texSource, x - 2.0f * dx, y);
    t0 -= SAMPLE(texSource, x - 1.0f * dx, y) * 8.0f;
    t0 += SAMPLE(texSource, x + 1.0f * dx, y) * 8.0f;
    t0 -= SAMPLE(texSource, x + 2.0f * dx, y);
    t0 /= 12.0f;

    t1 = SAMPLE(texTarget, x - 2.0f * dx, y);
    t1 -= SAMPLE(texTarget, x - 1.0f * dx, y) * 8.0f;
    t1 += SAMPLE(texTarget, x + 1.0f * dx, y) * 8.0f;
    t1 -= SAMPLE(texTarget, x + 2.0f * dx, y);
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // t derivative
    Iz[pos] = SAMPLE(texTarget, x, y) - SAMPLE(texSource, x, y);

    // y derivative
    t0 = SAMPLE(texSource, x, y - 2.0f * dy);
    t0 -= SAMPLE(texSource, x, y - 1.0f * dy) * 8.0f;
    t0 += SAMPLE(texSource, x, y + 1.0f * dy) * 8.0f;
    t0 -= SAMPLE(texSource, x, y + 2.0f * dy);
    t0 /= 12.0f;

    t1 = SAMPLE(texTarget, x, y - 2.0f * dy);
    t1 -= SAMPLE(texTarget, x, y - 1.0f * dy) * 8.0f;
    t1 += SAMPLE(texTarget, x, y + 1.0f * dy) * 8.0f;
    t1 -= SAMPLE(texTarget, x, y + 2.0f * dy);
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;

#undef SAMPLE
}

#if !defined(__HIP_NO_IMAGE_SUPPORT)
///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives (hardware texture path)
///
/// Used on devices with hardware texture units (e.g. RDNA).
/// \param[in]  width     image width
/// \param[in]  height    image height
/// \param[in]  stride    image stride in elements
/// \param[out] Ix        x derivative
/// \param[out] Iy        y derivative
/// \param[out] Iz        temporal derivative
/// \param[in]  texSource source image texture object
/// \param[in]  texTarget target image texture object
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernel(int                width,
                                         int                height,
                                         int                stride,
                                         float             *Ix,
                                         float             *Iy,
                                         float             *Iz,
                                         hipTextureObject_t texSource,
                                         hipTextureObject_t texTarget)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

#define SAMPLE(tex, u, v) tex2D<float>(tex, u, v)

    float t0, t1;
    // x derivative
    t0 = SAMPLE(texSource, x - 2.0f * dx, y);
    t0 -= SAMPLE(texSource, x - 1.0f * dx, y) * 8.0f;
    t0 += SAMPLE(texSource, x + 1.0f * dx, y) * 8.0f;
    t0 -= SAMPLE(texSource, x + 2.0f * dx, y);
    t0 /= 12.0f;

    t1 = SAMPLE(texTarget, x - 2.0f * dx, y);
    t1 -= SAMPLE(texTarget, x - 1.0f * dx, y) * 8.0f;
    t1 += SAMPLE(texTarget, x + 1.0f * dx, y) * 8.0f;
    t1 -= SAMPLE(texTarget, x + 2.0f * dx, y);
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // t derivative
    Iz[pos] = SAMPLE(texTarget, x, y) - SAMPLE(texSource, x, y);

    // y derivative
    t0 = SAMPLE(texSource, x, y - 2.0f * dy);
    t0 -= SAMPLE(texSource, x, y - 1.0f * dy) * 8.0f;
    t0 += SAMPLE(texSource, x, y + 1.0f * dy) * 8.0f;
    t0 -= SAMPLE(texSource, x, y + 2.0f * dy);
    t0 /= 12.0f;

    t1 = SAMPLE(texTarget, x, y - 2.0f * dy);
    t1 -= SAMPLE(texTarget, x, y - 1.0f * dy) * 8.0f;
    t1 += SAMPLE(texTarget, x, y + 1.0f * dy) * 8.0f;
    t1 -= SAMPLE(texTarget, x, y + 2.0f * dy);
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;

#undef SAMPLE
}
#endif // !__HIP_NO_IMAGE_SUPPORT

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  w            image width
/// \param[in]  h            image height
/// \param[in]  s            image stride
/// \param[out] Ix           x derivative
/// \param[out] Iy           y derivative
/// \param[out] Iz           temporal derivative
/// \param[in]  imageSupport non-zero if the current device has image support
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0,
                               const float *I1,
                               int          w,
                               int          h,
                               int          s,
                               float       *Ix,
                               float       *Iy,
                               float       *Iz,
                               int          imageSupport)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    if (!imageSupport)
    {
        ComputeDerivativesKernelSW<<<blocks, threads>>>(w, h, s, Ix, Iy, Iz, I0, w, h, s, I1);
        HIP_CHECK(hipGetLastError());
        return;
    }

#if !defined(__HIP_NO_IMAGE_SUPPORT)
    hipTextureDesc texDescr        = {};
    texDescr.normalizedCoords      = true;
    texDescr.filterMode            = hipFilterModeLinear;
    texDescr.addressMode[0]        = hipAddressModeMirror;
    texDescr.addressMode[1]        = hipAddressModeMirror;
    texDescr.readMode              = hipReadModeElementType;

    hipResourceDesc texRes               = {};
    texRes.resType                       = hipResourceTypePitch2D;
    texRes.res.pitch2D.desc              = hipCreateChannelDesc<float>();
    texRes.res.pitch2D.width             = w;
    texRes.res.pitch2D.height            = h;
    texRes.res.pitch2D.pitchInBytes      = s * sizeof(float);

    hipTextureObject_t texSource, texTarget;
    texRes.res.pitch2D.devPtr = (void *)I0;
    HIP_CHECK(hipCreateTextureObject(&texSource, &texRes, &texDescr, NULL));
    texRes.res.pitch2D.devPtr = (void *)I1;
    HIP_CHECK(hipCreateTextureObject(&texTarget, &texRes, &texDescr, NULL));

    ComputeDerivativesKernel<<<blocks, threads>>>(w, h, s, Ix, Iy, Iz, texSource, texTarget);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipDestroyTextureObject(texSource));
    HIP_CHECK(hipDestroyTextureObject(texTarget));
#endif
}
