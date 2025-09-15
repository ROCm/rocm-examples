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

#include "dataset.hpp"

#include "projection.hpp"
#include "utility.hpp"
#include "volume.hpp"

#include <tiffio.h>
#include <tiffio.hxx>

#include <cmath>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

namespace dataset
{
    auto make_projectionGeometry(std::string path) noexcept(false) -> projectionGeometry
    {
        std::ifstream file{path.c_str(), std::ios::binary};
        if(!file.is_open())
            throw std::runtime_error{"Could not open " + path + " for reading."};

        auto tiff = TIFFStreamOpen(path.c_str(), &file);
        if(tiff == nullptr)
            throw std::runtime_error{"Could not open TIFF file at " + path + " for reading."};

        auto N_h = 0u;
        auto N_v = 0u;
        get_field(tiff, TIFFTAG_IMAGEWIDTH, N_h, "Could not determine TIFF's image width.");
        get_field(tiff, TIFFTAG_IMAGELENGTH, N_v, "Could not determine TIFF's image length.");
        TIFFClose(tiff);

        // Filter length
        auto N_hFFT = std::pow(2.f, std::ceil(std::log2(N_h)));
        auto s_N_hFFT = static_cast<int>(N_hFFT);

        // Transformed filter length
        auto N_hTrans = N_hFFT / 2 + 1;

        constexpr auto d_h = 0.05f;
        constexpr auto d_v = 0.05f;

        constexpr auto shift_h = -4;
        constexpr auto shift_v = 0;

        constexpr auto delta_h = shift_h * d_h;
        constexpr auto delta_v = shift_v * d_v;

        // N_h: number of detector pixels (horizontal)
        // d_h: detector pixel size (mm)
        // delta_h: physical shift (mm), negative values shift detector coordinate system to the left
        auto h_min = -((N_h - 1) * d_h) / 2.f + delta_h;
        auto v_min = -((N_v - 1) * d_v) / 2.f + delta_v;

        return projectionGeometry
        {
            553.74f,                  // d_sd
            210.66f,                  // d_so
            float2{d_h, d_v},         // pixelDim
            0.5f,                     // thetaStep
            -1.f,                     // thetaSign
            int2{shift_h, shift_v},   // shift
            float2{delta_h, delta_v}, // delta
            12,                       // bps
            720,                      // num_proj
            uint2{N_h, N_v},          // dim,
            uint2{N_hFFT, N_v},       // dimFFT
            int2{s_N_hFFT, N_v},      // s_dimFFT
            uint2{N_hTrans, N_v},     // dimTrans
            float2{h_min, v_min}
        };
    }

    auto make_volumeGeometry(projectionGeometry const& projGeom) noexcept -> volumeGeometry
    {
        // Due to the circular path that the source-detector arrangement travels during exposure, the outer rays of the
        // cone form the tangents of a circle with radius "radius", starting from the center of rotation. This circle is
        // the support area for the horizontal x-y-plane. "radius" can be determined via the angle "alpha", which in
        // turn can be determined from the ratio of half the detector width to the source-detector distance.
        auto const alpha = std::atan(((projGeom.dim.x * projGeom.pixelDim.x) / 2.f) / projGeom.d_sd);
        auto const radius = std::abs(projGeom.d_so) * std::sin(alpha);

        auto const d_x = radius / (
                (((projGeom.dim.x * projGeom.pixelDim.x) / 2.f) + 
                 std::abs(projGeom.delta.x)) / projGeom.pixelDim.x
        );
        auto const d_y = d_x;
        auto const d_z = d_x;

        auto const N_x = (2.f * radius) / d_x;
        auto const N_y = N_x;
        auto const N_z = ((projGeom.dim.y * projGeom.pixelDim.y / 2.f) +
                          std::abs(projGeom.delta.y)) * (std::abs(projGeom.d_so) / projGeom.d_sd) * (2.f / d_z);

        return volumeGeometry{float3{d_x, d_y, d_z}, ulonglong3{N_x, N_y, N_z}};
    }
}
