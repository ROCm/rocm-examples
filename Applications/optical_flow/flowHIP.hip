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

#include "common.h"

// include kernels
#include "addKernel.hip"
#include "derivativesKernel.hip"
#include "downscaleKernel.hip"
#include "solverKernel.hip"
#include "upscaleKernel.hip"
#include "warpingKernel.hip"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowHIP(const float *I0,
                     const float *I1,
                     int          width,
                     int          height,
                     int          stride,
                     float        alpha,
                     int          nLevels,
                     int          nWarpIters,
                     int          nSolverIters,
                     float       *u,
                     float       *v)
{
    printf("Computing optical flow on GPU...\n");

    // pI0 and pI1 will hold device pointers
    float **pI0 = new float *[nLevels];
    float **pI1 = new float *[nLevels];

    int *pW = new int[nLevels];
    int *pH = new int[nLevels];
    int *pS = new int[nLevels];

    // device memory pointers
    float *d_tmp;
    float *d_du0;
    float *d_dv0;
    float *d_du1;
    float *d_dv1;

    float *d_Ix;
    float *d_Iy;
    float *d_Iz;

    float *d_u;
    float *d_v;
    float *d_nu;
    float *d_nv;

    const int dataSize = stride * height * sizeof(float);

    HIP_CHECK(hipMalloc(&d_tmp, dataSize));
    HIP_CHECK(hipMalloc(&d_du0, dataSize));
    HIP_CHECK(hipMalloc(&d_dv0, dataSize));
    HIP_CHECK(hipMalloc(&d_du1, dataSize));
    HIP_CHECK(hipMalloc(&d_dv1, dataSize));

    HIP_CHECK(hipMalloc(&d_Ix, dataSize));
    HIP_CHECK(hipMalloc(&d_Iy, dataSize));
    HIP_CHECK(hipMalloc(&d_Iz, dataSize));

    HIP_CHECK(hipMalloc(&d_u, dataSize));
    HIP_CHECK(hipMalloc(&d_v, dataSize));
    HIP_CHECK(hipMalloc(&d_nu, dataSize));
    HIP_CHECK(hipMalloc(&d_nv, dataSize));

    // prepare pyramid

    int currentLevel = nLevels - 1;
    // allocate GPU memory for input images
    HIP_CHECK(hipMalloc(&pI0[currentLevel], dataSize));
    HIP_CHECK(hipMalloc(&pI1[currentLevel], dataSize));

    HIP_CHECK(hipMemcpy(pI0[currentLevel], I0, dataSize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pI1[currentLevel], I1, dataSize, hipMemcpyHostToDevice));

    pW[currentLevel] = width;
    pH[currentLevel] = height;
    pS[currentLevel] = stride;

    for (; currentLevel > 0; --currentLevel) {
        int nw = pW[currentLevel] / 2;
        int nh = pH[currentLevel] / 2;
        int ns = iAlignUp(nw);

        HIP_CHECK(hipMalloc(&pI0[currentLevel - 1], ns * nh * sizeof(float)));
        HIP_CHECK(hipMalloc(&pI1[currentLevel - 1], ns * nh * sizeof(float)));

        Downscale(pI0[currentLevel],
                  pW[currentLevel],
                  pH[currentLevel],
                  pS[currentLevel],
                  nw,
                  nh,
                  ns,
                  pI0[currentLevel - 1]);

        Downscale(pI1[currentLevel],
                  pW[currentLevel],
                  pH[currentLevel],
                  pS[currentLevel],
                  nw,
                  nh,
                  ns,
                  pI1[currentLevel - 1]);

        pW[currentLevel - 1] = nw;
        pH[currentLevel - 1] = nh;
        pS[currentLevel - 1] = ns;
    }

    HIP_CHECK(hipMemset(d_u, 0, stride * height * sizeof(float)));
    HIP_CHECK(hipMemset(d_v, 0, stride * height * sizeof(float)));

    // compute flow
    for (; currentLevel < nLevels; ++currentLevel) {
        for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
            HIP_CHECK(hipMemset(d_du0, 0, dataSize));
            HIP_CHECK(hipMemset(d_dv0, 0, dataSize));

            HIP_CHECK(hipMemset(d_du1, 0, dataSize));
            HIP_CHECK(hipMemset(d_dv1, 0, dataSize));

            // on current level we compute optical flow
            // between frame 0 and warped frame 1
            WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel], pS[currentLevel], d_u, d_v, d_tmp);

            ComputeDerivatives(
                pI0[currentLevel], d_tmp, pW[currentLevel], pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

            for (int iter = 0; iter < nSolverIters; ++iter) {
                SolveForUpdate(d_du0,
                               d_dv0,
                               d_Ix,
                               d_Iy,
                               d_Iz,
                               pW[currentLevel],
                               pH[currentLevel],
                               pS[currentLevel],
                               alpha,
                               d_du1,
                               d_dv1);

                Swap(d_du0, d_du1);
                Swap(d_dv0, d_dv1);
            }

            // update u, v
            Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
            Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
        }

        if (currentLevel != nLevels - 1) {
            // prolongate solution
            float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];

            Upscale(d_u,
                    pW[currentLevel],
                    pH[currentLevel],
                    pS[currentLevel],
                    pW[currentLevel + 1],
                    pH[currentLevel + 1],
                    pS[currentLevel + 1],
                    scaleX,
                    d_nu);

            float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

            Upscale(d_v,
                    pW[currentLevel],
                    pH[currentLevel],
                    pS[currentLevel],
                    pW[currentLevel + 1],
                    pH[currentLevel + 1],
                    pS[currentLevel + 1],
                    scaleY,
                    d_nv);

            Swap(d_u, d_nu);
            Swap(d_v, d_nv);
        }
    }

    HIP_CHECK(hipMemcpy(u, d_u, dataSize, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(v, d_v, dataSize, hipMemcpyDeviceToHost));

    // cleanup
    for (int i = 0; i < nLevels; ++i) {
        HIP_CHECK(hipFree(pI0[i]));
        HIP_CHECK(hipFree(pI1[i]));
    }

    delete[] pI0;
    delete[] pI1;
    delete[] pW;
    delete[] pH;
    delete[] pS;

    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_du0));
    HIP_CHECK(hipFree(d_dv0));
    HIP_CHECK(hipFree(d_du1));
    HIP_CHECK(hipFree(d_dv1));
    HIP_CHECK(hipFree(d_Ix));
    HIP_CHECK(hipFree(d_Iy));
    HIP_CHECK(hipFree(d_Iz));
    HIP_CHECK(hipFree(d_nu));
    HIP_CHECK(hipFree(d_nv));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_v));
}
