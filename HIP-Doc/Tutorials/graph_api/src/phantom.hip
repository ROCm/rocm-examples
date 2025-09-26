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

#include "phantom.hpp"

#include "projection.hpp"
#include "utility.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

struct spheroid
{
    float x;          // Spheroid center (x axis), relative to origin
    float y;          // Spheroid center (y axis), relative to origin
    float horizontal; // horizontal axis of the original phantom (also the spheroids rotation axis)
    float vertical;   // vertical axis of the original phantom
    float theta;      // inclination angle (degrees)
    float intensity;  // intensity value (1 = black, 2 = white)
};

constexpr spheroid shepp_logan_phantom[] =
{
    // x,      y,       minor,   major,   theta, intensity
    {  0.f,    0.f,     0.69f,   0.92f,   0.f,   2.f},      // Great white spheroid
    {  0.f,   -0.0184f, 0.6624f, 0.874f,  0.f,  -0.98f},    // Great grey spheroid
    {  0.22f,  0.f,     0.11f,   0.31f,  -18.f, -0.02f},    // Right black spheroid
    { -0.22f,  0.f,     0.16f,   0.41f,   18.f, -0.02f},    // Left black spheroid
    {  0.f,    0.35f,   0.21f,   0.25f,   0.f,   0.01f},    // Top spheroid
    {  0.f,    0.1f,    0.046f,  0.046f,  0.f,   0.01f},    // Upper small spheroid
    {  0.f,   -0.1f,    0.046f,  0.046f,  0.f,   0.01f},    // Lower small spheroid
    { -0.08f, -0.605f,  0.046f,  0.023f,  0.f,   0.01f},    // Lower left tiny spheroid
    {  0.f,   -0.605f,  0.023f,  0.023f,  0.f,   0.01f},    // Lower middle tiny spheroid
    {  0.06f, -0.605f,  0.023f,  0.023f,  0.f,   0.01f},    // Lower right tiny spheroid
};
constexpr auto num_spheroids = 10;

__global__ void create_phantom_kernel(float* vol, std::size_t pitch, ulonglong3 dim, float3 voxelDim)
{
    auto x_in = blockIdx.x * blockDim.x + threadIdx.x;
    auto y_in = blockIdx.y * blockDim.y + threadIdx.y;
    auto z_in = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_in >= dim.x || y_in >= dim.y || z_in >= dim.z)
        return;

    auto slice_pitch = pitch * dim.y;
    auto slice = reinterpret_cast<char*>(vol) + z_in * slice_pitch;
    auto row = reinterpret_cast<float*>(slice + y_in * pitch);

    // Since our spheroids are defined in normalized coordinates / values, we need to scale to the physical extents
    auto const denorm = float3
    {
        ((dim.x - 1) * voxelDim.x) / 2.f,
        ((dim.y - 1) * voxelDim.y) / 2.f,
        ((dim.z - 1) * voxelDim.z) / 2.f
    };

    // Compute physical coordinates (centered at origin)
    auto y_flipped = dim.y - 1 - y_in; // Flip y axis to match original Shepp-Logan phantom orientation
    auto x = (x_in - 0.5f * (dim.x - 1)) * voxelDim.x;
    auto y = (y_flipped - 0.5f * (dim.y - 1)) * voxelDim.y;
    auto z = (z_in - 0.5f * (dim.z - 1)) * voxelDim.z;

    auto val = 0.f;

    // Add each spheroid's contribution
    for (auto e = 0; e < num_spheroids; ++e)
    {
        // Translate to spheroid center
        auto xt = x - (shepp_logan_phantom[e].x * denorm.x);
        auto yt = y - (shepp_logan_phantom[e].y * denorm.y);

        // Rotate in xy-plane by -theta
        auto theta_rad = -shepp_logan_phantom[e].theta * std::numbers::pi_v<float> / 180.0f;
        auto xtr = xt * cosf(theta_rad) - yt * sinf(theta_rad);
        auto ytr = xt * sinf(theta_rad) + yt * cosf(theta_rad);

        // Denormalize axis
        auto horizontal = shepp_logan_phantom[e].horizontal * denorm.x;
        auto vertical = shepp_logan_phantom[e].vertical * denorm.y;

        // Are we inside the spheroid? (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
        auto inside =
            (xtr / horizontal) * (xtr / horizontal) + 
            (ytr / vertical) * (ytr / vertical) + 
            (z / horizontal) * (z / horizontal) <= 1.f;

        if (inside)
            val += shepp_logan_phantom[e].intensity;
    }

    // Normalize final value to [0, 1]: in the original scale, 1 is black and 2 is white
    if(val > 0.f)
        val -= 1.f;

    row[x_in] = val;
}

auto create_phantom(ulonglong3 dim, float3 voxelDim) noexcept(false) -> hipPitchedPtr
{
    create_volume("shepp_logan_phantom.tif");

    auto hostVolPtr = static_cast<float*>(nullptr);
    hip_check(hipHostMalloc(&hostVolPtr, dim.x * dim.y * dim.z * sizeof(float), hipHostMallocDefault));
    auto hostVol = make_hipPitchedPtr(hostVolPtr, dim.x * sizeof(float), dim.x, dim.y);

    auto vol = hipPitchedPtr{};
    auto volExt = make_hipExtent(dim.x * sizeof(float), dim.y, dim.z);
    hip_check(hipMalloc3D(&vol, volExt));
    hip_check(hipMemset3D(vol, 0, volExt));

    auto blockSize = dim3{32u, 8u, 4u};
    auto blocks = dim3{
        static_cast<std::uint32_t>(dim.x + blockSize.x - 1) / blockSize.x,
        static_cast<std::uint32_t>(dim.y + blockSize.y - 1) / blockSize.y,
        static_cast<std::uint32_t>(dim.z + blockSize.z - 1) / blockSize.z
    };

    create_phantom_kernel<<<blocks, blockSize>>>(static_cast<float*>(vol.ptr), vol.pitch, dim, voxelDim);
    
    auto memcpy_params = hipMemcpy3DParms{};
    std::memset(&memcpy_params, 0, sizeof(hipMemcpy3DParms));
    memcpy_params.dstPos = make_hipPos(0, 0, 0);
    memcpy_params.dstPtr = hostVol;
    memcpy_params.srcPos = make_hipPos(0, 0, 0);
    memcpy_params.srcPtr = vol;
    memcpy_params.extent = volExt;
    memcpy_params.kind = hipMemcpyDeviceToHost;
    hip_check(hipMemcpy3D(&memcpy_params));

    auto save_vol_args = new save_volume_args
    {
        "shepp_logan_phantom.tif",
        hostVolPtr,
        dim.x,
        dim.y,
        dim.z,
        voxelDim.x,
        voxelDim.y
    };
    save_volume(static_cast<void*>(save_vol_args));

    hip_check(hipFreeHost(hostVolPtr));

    return vol;
}

__device__ auto siddon_raytrace(float const* vol,
                                std::size_t volPitch,
                                ulonglong3 volDims_vec,
                                float3 voxelDims_vec,
                                float3 sourcePos_vec,
                                float3 pixelPos) -> float
{
    unsigned long long volDims[3] = {volDims_vec.x, volDims_vec.y, volDims_vec.z};
    float voxelDims[3] = {voxelDims_vec.x, voxelDims_vec.y, voxelDims_vec.z};
    float sourcePos[3] = {sourcePos_vec.x, sourcePos_vec.y, sourcePos_vec.z};

    // Compute ray direction and length
    auto r_vec = pixelPos - sourcePos_vec;
    auto r_len = sqrtf(r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z);
    float r[3] = {r_vec.x, r_vec.y, r_vec.z};

    auto alpha_min = 0.f;                   // Entry parameter with volume
    auto alpha_max = 1.f;                   // Exit parameter with volume
    float alpha_next[3] = {0.f, 0.f, 0.f};  // value of alpha at which the ray will cross a voxel boundary along axis
    float alpha_inc[3] = {0.f, 0.f, 0.f};   // increment for alpha_next when stepping to next voxel along axis
    
    std::int64_t i[3] = {0, 0, 0};     // Current voxel index along each axis
    int i_step[3] = {0, 0, 0};         // direction of voxel stepping: -1 backward, 0 don't step, 1 forward

    // Set volume boundaries
    float boundsMin[3] =
    {
        -0.5f * (volDims[0] - 1.f) * voxelDims[0],
        -0.5f * (volDims[1] - 1.f) * voxelDims[1],
        -0.5f * (volDims[2] - 1.f) * voxelDims[2]
    };
    float boundsMax[3] = 
    {
        0.5f * (volDims[0] - 1.f) * voxelDims[0],
        0.5f * (volDims[1] - 1.f) * voxelDims[1],
        0.5f * (volDims[2] - 1.f) * voxelDims[2]
    };

    // Calculate ray intersection with the volume and stepping parameters
    for(auto axis = 0; axis < 3; ++axis)
    {
        auto s0 = sourcePos[axis]; // Ray start coordinate for current axis
        auto r_axis = r[axis];  // Ray direction component for current axis

        if(fabsf(r_axis) < std::numeric_limits<float>::epsilon()) // Ray parallel to axis
        {
            // Is the ray outside of the volume? If so, there is no intersection
            if(s0 < boundsMin[axis] || s0 > boundsMax[axis])
                return 0.f;

            // parallel to axis; will never cross voxel boundary
            alpha_next[axis] = std::numeric_limits<float>::max();
            alpha_inc[axis] = std::numeric_limits<float>::max();
            i_step[axis] = 0;
        }
        else
        {
            // Compute alpha for entry/exit of volume slabs in this axis
            auto alpha1 = (boundsMin[axis] - s0) / r_axis;
            auto alpha2 = (boundsMax[axis] - s0) / r_axis;
            auto alphaE = fminf(alpha1, alpha2);
            auto alphaX = fmaxf(alpha1, alpha2);
            alpha_min = fmaxf(alpha_min, alphaE);
            alpha_max = fminf(alpha_max, alphaX);

            // Determine stepping direction
            i_step[axis] = (r_axis > 0) ? 1 : -1;

            // Find starting voxel boundary along the axis
            auto pos = s0 + alpha_min * r_axis;

            // Find initial voxel index
            // Account for last row's rays starting at the detector boundary
            auto tolerance = 0.5f * std::numeric_limits<float>::epsilon();
            i[axis] = static_cast<long long>(floorf((pos - boundsMin[axis]) / voxelDims[axis])) + tolerance;
            i[axis] = max(0ll, min(volDims[axis] - 1, i[axis]));
            if(i_step[axis] > 0) // Next boundary is "upper" face of the voxel
                alpha_next[axis] = (boundsMin[axis] + (i[axis] + 1) * voxelDims[axis] - s0) / r_axis;
            else // Next boundary is "lower" face of the voxel
                alpha_next[axis] = (boundsMin[axis] + i[axis] * voxelDims[axis] - s0) / r_axis;

            // Distance between crossings along current axis
            alpha_inc[axis] = voxelDims[axis] / fabsf(r_axis);
        }
    }

    // Check if the ray actually intersects the volume
    if(alpha_min > alpha_max)
        return 0.f;

    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    // Step through voxels along trajectory
    auto alpha_current = alpha_min;
    auto sum = 0.f;
    while(alpha_current < alpha_max)
    {
        // Which axis will the ray cross next?
        auto min_axis = 0ll;
        auto alpha_next_min = alpha_next[0];
        for(auto axis = 1; axis < 3; ++axis)
        {
            if(alpha_next[axis] < alpha_next_min)
            {
                alpha_next_min = alpha_next[axis];
                min_axis = axis;
            }
        }

        // Either hit next voxel boundary or leave volume
        auto alpha_exit = fminf(alpha_next_min, alpha_max);
        auto d_alpha = alpha_exit - alpha_current;

        // Get current voxel value and accumulate weighted by intersection length
        auto val = float{};
        if(i[0] >= 0ll && i[0] < volDims[0] && i[1] >= 0ll && i[1] < volDims[1] && i[2] >= 0ll && i[2] < volDims[2])
        {
            auto slicePitch = volPitch * volDims[1];
            auto slice = reinterpret_cast<char const*>(vol) + i[2] * slicePitch;
            auto row = reinterpret_cast<float const*>(slice + i[1] * volPitch);
            val = row[i[0]];
        }

        // Convert d_alpha to physical distance
        auto physicalDistance = d_alpha * r_len;
        sum += val * physicalDistance;
        
        alpha_current = alpha_exit;
        if(alpha_current > alpha_max)
            break;

        // Move to next voxel along the axis crossed
        i[min_axis] += i_step[min_axis];
        alpha_next[min_axis] += alpha_inc[min_axis];

        // If voxel index is out of bounds, exit
        if(i[min_axis] < 0 || i[min_axis] >= volDims[min_axis])
            break;
    }

    return sum;
}

__global__ void create_projection_kernel(
    float const* __restrict__ vol,
    std::size_t volPitch,
    ulonglong3 volDims,
    float3 voxelDims,
    std::uint16_t* __restrict__ proj,
    std::size_t projPitch,
    uint2 projDim,
    float2 pixelDims,
    float d_so,
    float d_sd,
    float3 sourcePos,
    float3 originPos,
    float3 detectorCenter,
    float3 h_axis,
    float3 v_axis
)
{
    int h_in = blockIdx.x * blockDim.x + threadIdx.x;
    int v_in = blockIdx.y * blockDim.y + threadIdx.y;

    if (h_in >= projDim.x || v_in >= projDim.y)
        return;

    // Get detector pixel position in space
    auto pixelPos = detectorCenter +
                    (h_in - projDim.x / 2.f + 0.5f) * pixelDims.x * h_axis +
                    (v_in - projDim.y / 2.f + 0.5f) * pixelDims.y * v_axis;

    // Raytrace from source to pixel (Siddon's algorithm)
    auto attenuation = siddon_raytrace(vol, volPitch, volDims, voxelDims, sourcePos, pixelPos);

    // Convert to transmission value (Beer-Lambert law)
    auto transmission = expf(-attenuation);

    // Store as 12-bit unsigned integer
    auto val = static_cast<std::uint16_t>(roundf(transmission * 4095.f));

    // Save
    auto row = reinterpret_cast<std::uint16_t*>(reinterpret_cast<char*>(proj) + v_in * projPitch);
    row[h_in] = val;
}

void create_projection(
    hipPitchedPtr vol,
    volumeGeometry const& volGeom,
    hipPitchedPtr proj,
    projectionGeometry const& projGeom,
    std::uint32_t projIdx
) noexcept
{
    auto theta = projGeom.thetaSign * projGeom.thetaStep * projIdx;
    auto thetaRad = theta * std::numbers::pi_v<float> / 180.0f;

    auto originPos = float3{0.f, 0.f, 0.f};
    auto sourcePos = float3{-projGeom.d_so * std::cos(thetaRad), -projGeom.d_so * std::sin(thetaRad), 0.f};

    // Direction vector from source to origin
    auto direction = originPos - sourcePos;
    auto direction_len = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction /= direction_len;

    // Place detector center
    auto detectorCenter = sourcePos + direction * projGeom.d_sd;

    // Get detector orientation
    // Horizontal axis (perpendicular to ray direction in xy-plane)
    auto h_axis = float3{-std::sin(thetaRad), std::cos(thetaRad), 0.f};
    auto h_axis_len = std::sqrt(h_axis.x * h_axis.x + h_axis.y * h_axis.y + h_axis.z * h_axis.z);
    h_axis /= h_axis_len;

    // Vertical axis (perpendicular to ray direction and horizontal axis)
    auto v_axis = float3
    {
        direction.y * h_axis.z - direction.z * h_axis.y,
        direction.z * h_axis.x - direction.x * h_axis.z,
        direction.x * h_axis.y - direction.y * h_axis.x
    };
    auto v_axis_len = std::sqrt(v_axis.x * v_axis.x + v_axis.y * v_axis.y + v_axis.z * v_axis.z);
    v_axis /= v_axis_len;

    auto threadsPerBlock = dim3{};
    threadsPerBlock.x = std::min(projGeom.dim.x, 1024u);
    threadsPerBlock.y = 1024u / threadsPerBlock.x;
    threadsPerBlock.z = 1u;

    auto blocksPerGrid = dim3{};
    blocksPerGrid.x = (projGeom.dim.x % threadsPerBlock.x == 0u) ? projGeom.dim.x / threadsPerBlock.x :
                                                                   projGeom.dim.x / threadsPerBlock.x + 1u;
    blocksPerGrid.y = (projGeom.dim.y % threadsPerBlock.y == 0u) ? projGeom.dim.y / threadsPerBlock.y :
                                                                   projGeom.dim.y / threadsPerBlock.y + 1u;
    blocksPerGrid.z = 1u;

    create_projection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<float*>(vol.ptr),
        vol.pitch,
        volGeom.dim,
        volGeom.voxelDim,
        static_cast<std::uint16_t*>(proj.ptr),
        proj.pitch,
        projGeom.dim,
        projGeom.pixelDim,
        projGeom.d_so,
        projGeom.d_sd,
        sourcePos,
        originPos,
        detectorCenter,
        h_axis,
        v_axis
    );
}

namespace phantom
{
    auto make_projectionGeometry() noexcept -> projectionGeometry
    {
        constexpr auto N_h = 512u;
        constexpr auto N_v = 512u;
        
        constexpr auto d_h = 0.004f;
        constexpr auto d_v = 0.004f;

        // Filter length
        auto const N_hFFT = std::pow(2.f, std::ceil(std::log2(N_h)));
        auto const s_N_hFFT = static_cast<std::int32_t>(N_hFFT);

        // Transformed filter length
        auto const N_hTrans = N_hFFT / 2 + 1;

        constexpr auto h_min = -((N_h - 1) * d_h) / 2.f;
        constexpr auto v_min = -((N_v - 1) * d_v) / 2.f;

        constexpr auto d_so = 40.f;
        constexpr auto d_sd = d_so + 2.f;

        return projectionGeometry
        {
            d_sd,
            d_so,
            float2{d_h, d_v},            // pixelDim
            0.5f,                        // thetaStep
            1.f,                         // thetaSign
            int2{0, 0},                  // shift
            float2{0.f, 0.f},            // delta
            12,                          // bps
            720,                         // numProj            
            uint2{N_h, N_v},             // dim
            uint2{N_hFFT, N_v},          // dimFFT
            int2{s_N_hFFT, N_v},         // s_dimFFT
            uint2{N_hTrans, N_v},        // dimTrans
            float2{h_min, v_min}         // minCoord
        };
    }

    auto make_volumeGeometry() noexcept -> volumeGeometry
    {
        auto volumeDims = ulonglong3{512, 512, 512};
        auto voxelDims = float3{0.004, 0.004, 0.004};
        return volumeGeometry{voxelDims, volumeDims};
    };

    auto make_projections(
        projectionGeometry const& projGeom, volumeGeometry const& volGeom) noexcept(false) -> std::vector<hipPitchedPtr>
    {
        std::cout << "Generating dataset... ";

        auto volume = create_phantom(volGeom.dim, volGeom.voxelDim);

        auto projections = std::vector<hipPitchedPtr>{}; // Forward projections in uint12 format
        for(auto projIdx = 0; projIdx < projGeom.numProj; ++projIdx)
        {
            auto projPtr = static_cast<std::uint16_t*>(nullptr);
            auto projPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&projPtr), &projPitch, projGeom.dim.x * sizeof(float), projGeom.dim.y));
            auto proj = make_hipPitchedPtr(projPtr, projPitch, projGeom.dim.x, projGeom.dim.y);
            
            create_projection(volume, volGeom, proj, projGeom, projIdx);
            projections.push_back(proj);
        }

        hip_check(hipFree(volume.ptr));

        std::cout << "Done!" << std::endl;
        return projections;
    }
}

