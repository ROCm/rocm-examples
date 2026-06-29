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

#include <hip/hip_cooperative_groups.h>

#include "common.h"

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, HIP kernel.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// Template parameters are describe CTA size
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
template <int bx, int by>
__global__ void JacobiIteration(const float *du0,
                                const float *dv0,
                                const float *Ix,
                                const float *Iy,
                                const float *Iz,
                                int          w,
                                int          h,
                                int          s,
                                float        alpha,
                                float       *du1,
                                float       *dv1)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    volatile __shared__ float du[(bx + 2) * (by + 2)];
    volatile __shared__ float dv[(bx + 2) * (by + 2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

    // Load data to shared memory.
    // load tile being processed
    du[shMemPos] = du0[pos];
    dv[shMemPos] = dv0[pos];

    // load necessary neighbouring elements
    // We clamp out-of-range coordinates.
    // It is equivalent to mirroring
    // because we access data only one step away from borders.
    if (threadIdx.y == 0) {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        x = min(bsx + threadIdx.x, w - 1);
        // row just below the tile
        y         = max(bsy - 1, 0);
        gmPos     = y * s + x;
        smPos     = threadIdx.x + 1;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];

        // row above the tile
        y = min(bsy + by, h - 1);
        smPos += (by + 1) * (bx + 2);
        gmPos     = y * s + x;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
    }
    else if (threadIdx.y == 1) {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        y = min(bsy + threadIdx.x, h - 1);
        // column to the left
        x     = max(bsx - 1, 0);
        smPos = bx + 2 + threadIdx.x * (bx + 2);
        gmPos = x + y * s;

        // check if we are within tile
        if (threadIdx.x < by) {
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            // column to the right
            x     = min(bsx + bx, w - 1);
            gmPos = y * s + x;
            smPos += bx + 1;
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
        }
    }

    cg::sync(cta);

    if (ix >= w || iy >= h)
        return;

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + bx + 2;
    down  = shMemPos - bx - 2;

    float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
    float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

    float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = sumU - Ix[pos] * frac;
    dv1[pos] = sumV - Iy[pos] * frac;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, HIP kernel wrapper.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
static void SolveForUpdate(const float *du0,
                           const float *dv0,
                           const float *Ix,
                           const float *Iy,
                           const float *Iz,
                           int          w,
                           int          h,
                           int          s,
                           float        alpha,
                           float       *du1,
                           float       *dv1)
{
    // CTA size
    dim3 threads(32, 6);
    // grid size
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    JacobiIteration<32, 6><<<blocks, threads>>>(du0, dv0, Ix, Iy, Iz, w, h, s, alpha, du1, dv1);
}
