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

#include <tiffio.h>

#include <cstdarg>
#include <cstdio>

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
