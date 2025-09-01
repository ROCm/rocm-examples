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

#ifndef VOLUME_HPP
#define VOLUME_HPP

#include "projection.hpp"

#include <cstddef>
#include <string>

struct volume_geometry
{
    // Due to the circular path that the source-detector arrangement travels during exposure, the outer rays of the cone
    // form the tangents of a circle with radius "radius", starting from the center of rotation. This circle is the
    // support area for the horizontal x-y-plane. "radius" can be determined via the angle "alpha", which in turn can be
    // determined from the ratio of half the detector width to the source-detector distance.
    float alpha{};
    float radius{};
    
    float d_x{}; // physical voxel size in x direction [mm]
    float d_y{}; // physical voxel size in y direction [mm]
    float d_z{}; // physical voxel size in z direction [mm]
    std::size_t N_x{}; // number of voxels in x direction
    std::size_t N_y{}; // number of voxels in y direction
    std::size_t N_z{}; // number of voxels in z direction

    volume_geometry() noexcept = default;
    volume_geometry(projection_geometry const& proj_geom) noexcept;
};

void create_volume(std::string path) noexcept(false);

struct save_volume_args
{
    std::string path;
    float* vol;
    std::size_t N_x;
    std::size_t N_y;
    std::size_t N_z;
    std::size_t s_N_z;
    float d_x;
    float d_y;
};
void save_volume(void* args) noexcept;


#endif
