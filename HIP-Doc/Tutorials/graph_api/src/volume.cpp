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

#include "utility.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <tiffio.h>
#include <tiffio.hxx>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

void create_volume(std::string path) noexcept(false)
{
    auto tiff = TIFFOpen(path.c_str(), "w8"); // open in BigTIFF mode
    if(tiff == nullptr)
        throw std::runtime_error{"Could not open TIFF file at " + path + " for writing."};

    TIFFClose(tiff);
}

void save_volume(void* args) noexcept
{
    using namespace std::literals::string_literals;

    try
    {
        auto my_args = static_cast<save_volume_args*>(args);

        auto tiff = TIFFOpen(my_args->path.c_str(), "a8"); // open in BigTIFF mode
        if(tiff == nullptr)
            throw std::runtime_error{"Could not open TIFF file at " + my_args->path + " for writing."};

        auto set_field = [tiff](std::uint32_t tag, auto&& param, std::string error_msg)
        {
            if(auto err = TIFFSetField(tiff, tag, param); err != 1)
                throw std::runtime_error{error_msg};
        };

        for(auto slice_idx = 0; slice_idx < my_args->s_N_z; ++slice_idx)
        {
            if(slice_idx > 0u)
            {
                if(auto err = TIFFWriteDirectory(tiff); err != 1)
                    throw std::runtime_error{"Could not write directory entry to TIFF file."};
            }

            auto slice = my_args->vol + slice_idx * my_args->N_x * my_args->N_y;

            auto x_cm = 10.f / my_args->d_x;
            auto y_cm = 10.f / my_args->d_y;

            set_field(TIFFTAG_BITSPERSAMPLE, 8 * sizeof(float), "Could not set TIFF's bits per sample.");
            set_field(TIFFTAG_COMPRESSION, COMPRESSION_NONE, "Could not set TIFF's compression.");
            set_field(TIFFTAG_IMAGEDEPTH, my_args->N_z, "Could not set TIFF's image depth");
            set_field(TIFFTAG_IMAGELENGTH, my_args->N_y, "Could not set TIFF's image length.");
            set_field(TIFFTAG_IMAGEWIDTH, my_args->N_x, "Could not set TIFF's image width.");
            set_field(TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG, "Could not set TIFF's planar configuration");
            set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK, "Could not set TIFF's photometric interpretation.");
            set_field(TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER, "Could not set TIFF's resolution unit.");
            set_field(TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP, "Could not set TIFF's sample format.");
            set_field(TIFFTAG_SAMPLESPERPIXEL, 1, "Could not set TIFF's samples per pixel.");
            // set_field(TIFFTAG_SMAXSAMPLEVALUE, 1.f, "Could not set TIFF's maximum sample value.");
            // set_field(TIFFTAG_SMINSAMPLEVALUE, 0.f, "Could not set TIFF's minimum sample value.");
            set_field(TIFFTAG_XRESOLUTION, x_cm, "Could not set TIFF's horizontal resolution.");
            set_field(TIFFTAG_YRESOLUTION, y_cm, "Could not set TIFF's vertical resolution.");

            auto rows_per_strip = TIFFDefaultStripSize(tiff, 0);
            set_field(TIFFTAG_ROWSPERSTRIP, rows_per_strip, "Could not set TIFF's rows per strip.");

            auto strip_size = TIFFStripSize(tiff);
            if(strip_size == 0)
                throw std::runtime_error{"Could not obtain TIFF's strip size."};

            auto num_strips = TIFFNumberOfStrips(tiff);
            for(auto i = 0u; i < num_strips - 1; ++i)
            {
                auto const offset = i * strip_size / sizeof(float);
                if(auto err = TIFFWriteEncodedStrip(tiff, i, slice + offset, strip_size); err == -1)
                    throw std::runtime_error{"Could not write strip to TIFF. (slice "s +
                                             std::to_string(slice_idx) + ")"s};
            }

            auto const slice_size = my_args->N_x * my_args->N_y * sizeof(float);
            if(auto const remainder = slice_size % strip_size; remainder != 0)
                strip_size = remainder;
        
            auto strip_idx = num_strips - 1;
            auto const offset = strip_idx * strip_size / sizeof(float);
            if(auto err = TIFFWriteEncodedStrip(tiff, strip_idx, slice + offset, strip_size); err == -1)
                throw std::runtime_error{"Could not write strip to TIFF."};
        }

        TIFFClose(tiff);
        delete my_args;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Error saving volume: " << e.what() << std::endl;
        std::terminate();
    }
}
