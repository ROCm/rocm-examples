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

#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include "utility.hpp"

#include <hip/hip_runtime.h>

#include <tiffio.h>
#include <tiffio.hxx>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct projection_geometry
{
    // Detector constants - all units in mm unless stated otherwise
    static constexpr auto d_sd = 553.74f; // Distance between source and detector
    static constexpr auto d_so = 210.66f; // Distance between source and origin (object center).
    static constexpr auto d_h = 0.05f; // horizontal physical length of a pixel
    static constexpr auto d_v = 0.05f; // vertical physical length of a pixel
    static constexpr auto theta_step = 0.5f; // angle step size [°]
    static constexpr auto theta_sign = -1.f; // If the reconstructed volume shows ghosts / distortion, flip this
    static constexpr auto shift_h = -4; // horizontal shift [px]
    static constexpr auto shift_v = 0; // vertical shift [px]
    static constexpr auto delta_h = shift_h * d_h; // physical horizontal shift
    static constexpr auto delta_v = shift_v * d_v; // physical vertical shift
    static constexpr auto bps = 12; // bits per sample (pixel) in input projection [no unit]
    static constexpr auto num_proj = static_cast<std::uint32_t>(360.f / theta_step); // Total number of projections

    // All units are px unless stated otherwise
    std::uint32_t N_h{}; // horizontal size
    std::uint32_t N_v{}; // vertical size
    std::uint32_t N_hFFT{}; // filter length
    std::int32_t s_N_hFFT{}; // signed filter length
    std::uint32_t N_hTrans{}; // transformed length
    std::int32_t s_N_hTrans{}; // signed transformed length
    float h_min{}; // Horizontal physical center of the detector [mm]
    float v_min{}; // Vertical physical center of the detector [mm]

    projection_geometry() noexcept = default;
    projection_geometry(std::string path) noexcept(false);
};

struct load_projection_args
{
    std::string path;
    std::uint32_t N_h;
    std::uint32_t N_v;
    std::uint16_t* data;
};
void load_projection(void* args) noexcept;

struct save_projection_args
{
    std::string path;
    std::uint32_t N_h;
    std::uint32_t N_v;
    float* data;
};
template <typename T>
void save_projection(void* args) noexcept
{
    try
    {
        auto my_args = static_cast<save_projection_args*>(args);

        std::ofstream file{my_args->path.c_str(), std::ios::binary | std::ios::trunc};
        if(!file.is_open())
            throw std::runtime_error{"Could not open " + my_args->path + " for writing."};

        auto tiff = TIFFStreamOpen(my_args->path.c_str(), &file);
        if(tiff == nullptr)
            throw std::runtime_error{"Could not open TIFF file at " + my_args->path + " for writing."};

        auto set_field = [tiff](std::uint32_t tag, auto&& param, std::string error_msg)
        {
            if(auto err = TIFFSetField(tiff, tag, param); err != 1)
                throw std::runtime_error{error_msg};
        };

        set_field(TIFFTAG_IMAGEWIDTH, my_args->N_h, "Could not set TIFF's image width.");
        set_field(TIFFTAG_IMAGELENGTH, my_args->N_v, "Could not set TIFF's image length.");
        set_field(TIFFTAG_BITSPERSAMPLE, 8 * sizeof(T), "Could not set TIFF's bits per sample.");
        set_field(TIFFTAG_SAMPLESPERPIXEL, 1, "Could not set TIFF's samples per pixel.");
        set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK, "Could not set TIFF's photometric interpretation.");

        if constexpr(std::is_same_v<float, T>)
        {
            set_field(TIFFTAG_SMINSAMPLEVALUE, 0.f, "Could not set TIFF's minimum sample value.");
            set_field(TIFFTAG_SMAXSAMPLEVALUE, 1.f, "Could not set TIFF's maximum sample value.");
            set_field(TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP, "Could not set TIFF's sample format.");
        }
        else if constexpr(std::is_unsigned_v<T>)
            set_field(TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT, "Could not set TIFF's sample format.");

        auto rows_per_strip = TIFFDefaultStripSize(tiff, 0);
        set_field(TIFFTAG_ROWSPERSTRIP, rows_per_strip, "Could not set TIFF's rows per strip.");

        auto strip_size = TIFFStripSize(tiff);
        if(strip_size == 0)
            throw std::runtime_error{"Could not obtain TIFF's strip size."};

        auto num_strips = TIFFNumberOfStrips(tiff);
        for(auto i = 0u; i < num_strips - 1; ++i)
        {
            auto const offset = i * strip_size / sizeof(T);
            if(auto err = TIFFWriteRawStrip(tiff, i, my_args->data + offset, strip_size); err == -1)
                throw std::runtime_error{"Could not write strip to TIFF."};
        }

        auto const proj_size = my_args->N_h * my_args->N_v * sizeof(T);
        if(auto const remainder = proj_size % strip_size; remainder != 0)
            strip_size = remainder;
        
        auto strip_idx = num_strips - 1;
        auto const offset = strip_idx * strip_size / sizeof(T);
        if(auto err = TIFFWriteRawStrip(tiff, strip_idx, my_args->data + offset, strip_size); err == -1)
            throw std::runtime_error{"Could not write strip to TIFF."};

        TIFFClose(tiff);
        delete my_args;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Error saving projection: " << e.what() << std::endl;
        std::terminate();
    }
}

#endif
