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

#include "projection.hpp"

#include "utility.hpp"

#include <tiffio.h>
#include <tiffio.hxx>

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // Evil global variable
    auto logfile = static_cast<std::FILE*>(nullptr);

    [[gnu::constructor]] void init_libtiff()
    {
        auto handler = [](const char* /* module */, const char* fmt, std::va_list ap)
        {
            if(logfile == nullptr)
            {
                auto const path = std::tmpnam(nullptr);
                if(logfile = std::fopen(path, "w"); logfile == nullptr)
                {
                    std::perror("Could not open logfile for writing");
                    return;
                }   
            }
            
            std::vfprintf(logfile, fmt, ap);
            std::fprintf(logfile, "\n");
        };

        TIFFSetWarningHandler(handler);
    }

    [[gnu::destructor]] void shutdown_libtiff()
    {
        if(logfile == nullptr)
            return;
            
        if(auto err = std::fclose(logfile); err != 0)
            std::perror("Could not close logfile");
    }
}

void load_projection(void* args) noexcept
{
    try
    {
        auto my_args = static_cast<load_projection_args*>(args);
        std::ifstream file{my_args->path.c_str(), std::ios::binary};
        if(!file.is_open())
            throw std::runtime_error{"Could not open " + my_args->path + " for reading."};

        auto tiff = TIFFStreamOpen(my_args->path.c_str(), &file);
        if(tiff == nullptr)
            throw std::runtime_error{"Could not open TIFF file at " + my_args->path + " for reading."};

        auto strip_bytecounts = static_cast<std::uint64_t*>(nullptr);
        get_field(tiff, TIFFTAG_STRIPBYTECOUNTS, strip_bytecounts, "Could not determine TIFF's strip bytecounts.");

        auto strip_num = TIFFNumberOfStrips(tiff);
        for(auto i = 0u; i < strip_num; ++i)
        {
            auto const offset = i * strip_bytecounts[0] / sizeof(std::uint16_t);
            if(auto err = TIFFReadRawStrip(tiff, i, my_args->data + offset, strip_bytecounts[i]); err == -1)
                throw std::runtime_error{"Could not read strip from TIFF"};
        }

        TIFFClose(tiff);
        delete my_args;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Error loading projection: " << e.what() << std::endl;
        std::terminate();
    }
}
