// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_HIPRTC_UTILS_HPP
#define COMMON_HIPRTC_UTILS_HPP

#include <hip/hiprtc.h>
#include <vector>

/// \brief Checks if the provided error code is \p HIPRTC_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPRTC_CHECK(condition)                                                                \
    {                                                                                          \
        const hiprtcResult error = condition;                                                  \
        if(error != HIPRTC_SUCCESS)                                                            \
        {                                                                                      \
            std::cerr << "An error encountered: \"" << hiprtcGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                             \
            std::exit(error_exit_code);                                                        \
        }                                                                                      \
    }

static std::vector<char> compile_jit_callback(const std::string& src)
{
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, src.c_str(), "callback.hip", 0, nullptr, nullptr));

    std::vector<const char*> options;
#ifdef __HIP_PLATFORM_AMD__
    options.push_back("-O3");
    options.push_back("--offload-arch=amdgcnspirv");
#else
#ifdef CUDA_INCLUDE_DIR
    options.push_back("-I" CUDA_INCLUDE_DIR);
#endif
    options.push_back("-dlto");
    options.push_back("--relocatable-device-code=true");
#endif

    auto err = hiprtcCompileProgram(prog, options.size(), options.data());
    if(err != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            HIPRTC_CHECK(hiprtcGetProgramLog(prog, log.data()));
            std::cerr << "Compilation error:\n" << std::string(log.begin(), log.end()) << std::endl;
        }
        std::exit(err);
    }

    size_t            codeSize;
    std::vector<char> code;
#ifdef __HIP_PLATFORM_AMD__
    HIPRTC_CHECK(hiprtcGetBitcodeSize(prog, &codeSize));
    code.resize(codeSize);
    HIPRTC_CHECK(hiprtcGetBitcode(prog, code.data()));
#else
    auto nverr = nvrtcGetLTOIRSize(prog, &codeSize);
    if(nverr != NVRTC_SUCCESS)
      throw std::runtime_error{"failed to get bitcode size"};

    code.resize(codeSize);
    nverr = nvrtcGetLTOIR(prog, code.data());
    if(nverr != NVRTC_SUCCESS)
      throw std::runtime_error{"failed to get bitcode"};
#endif
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
    return code;
}

#endif
