// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_ROCPROFILER_UTILS_HPP
#define COMMON_ROCPROFILER_UTILS_HPP

#include "example_utils.hpp"

#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ROCPROFILER_ASSERT(condition, msg)                                                     \
    {                                                                                          \
        if(!(condition))                                                                       \
        {                                                                                      \
            std::cerr << "rocProfiler assertion failure: " << msg << " at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                                \
            abort();                                                                           \
        }                                                                                      \
    }

// Merged from Libraries/rocProfiler-SDK/common/filesystem.hpp
#if !defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM)
    #if defined __has_include
        #if __has_include(<ghc/filesystem.hpp>)
            #define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 1
        #else
            #define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 0
        #endif
    #else
        #define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 0
    #endif
#endif

#if ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM == 0
    #if defined __has_include
        #if __has_include(<version>)
            #include <version>
        #endif
    #endif

    #if defined(__cpp_lib_filesystem)
        #define ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM 1
    #else
        #if defined __has_include
            #if __has_include(<filesystem>)
                #define ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM 1
            #endif
        #endif
    #endif
#endif

// include the correct filesystem header
#if defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM) \
    && ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM > 0
    #include <ghc/filesystem.hpp>
#elif defined(ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM) \
    && ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM > 0
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif

/// \brief Checks if the provided rocProfiler status is \p ROCPROFILER_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCPROFILER_CHECK(condition)                                                          \
    {                                                                                         \
        const rocprofiler_status_t status = condition;                                        \
        if(status != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                     \
            std::cerr << "rocProfiler error encountered: \""                                  \
                      << rocprofiler_get_status_string(status) << "\" at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                               \
            std::exit(error_exit_code);                                                       \
        }                                                                                     \
    }

/// \brief Checks if the provided rocProfiler status is \p ROCPROFILER_STATUS_SUCCESS and if not,
/// prints an error message with custom message to the standard error output and terminates the
/// program with an error code.
#define ROCPROFILER_CALL(condition, msg)                                                   \
    {                                                                                      \
        const rocprofiler_status_t status = condition;                                     \
        if(status != ROCPROFILER_STATUS_SUCCESS)                                           \
        {                                                                                  \
            std::cerr << "[" #condition "][" << __FILE__ << ":" << __LINE__ << "] " << msg \
                      << " failed with error code " << status << ": "                      \
                      << rocprofiler_get_status_string(status) << std::endl;               \
            std::stringstream errmsg{};                                                    \
            errmsg << "[" #condition "][" << __FILE__ << ":" << __LINE__ << "] " << msg    \
                   << " failure (" << rocprofiler_get_status_string(status) << ")";        \
            throw std::runtime_error(errmsg.str());                                        \
        }                                                                                  \
    }

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y) ROCPROFILER_VAR_NAME_COMBINE(X, Y)

#define ROCPROFILER_WARN(result)                                                                \
    {                                                                                           \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;              \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)           \
        {                                                                                       \
            std::string status_msg                                                              \
                = rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));   \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << #result                  \
                      << " returned error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) \
                      << ": " << status_msg << ". This is just a warning!" << std::endl;        \
        }                                                                                       \
    }

#if HIP_VERSION >= 60300000
    #define HIP_HOST_ALLOC_FUNC hipHostMalloc
    #define HIP_HOST_FREE_FUNC hipHostFree
#else
    #define HIP_HOST_ALLOC_FUNC hipHostMalloc
    #define HIP_HOST_FREE_FUNC hipHostFree
#endif

namespace common
{
/// \brief Device information structure for HIP device management
struct device_info
{
    int device_count;
    int current_device;
};

/// \brief Performance timer class for measuring execution time
class performance_timer
{
private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::duration   elapsed_time_;

public:
    performance_timer()
    {
        reset_timer();
    }

    inline void reset_timer()
    {
        elapsed_time_ = std::chrono::steady_clock::duration(0);
    }

    inline void start_timer()
    {
        start_time_ = std::chrono::steady_clock::now();
    }

    inline void stop_timer()
    {
        const auto end_time = std::chrono::steady_clock::now();
        elapsed_time_ += end_time - start_time_;
    }

    /// @brief Returns time elapsed in seconds
    /// @return double that contains the elapsed time in seconds
    inline double get_elapsed_time() const
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_time_).count();
    }
};

/// \brief Initialize HIP devices and return device information
inline device_info initialize_hip_devices()
{
    device_info info = {};
    HIP_CHECK(hipGetDeviceCount(&info.device_count));

    if(info.device_count <= 0)
    {
        std::cerr << "No HIP devices found!" << std::endl;
        std::exit(error_exit_code);
    }

    return info;
}

/// \brief Set HIP device for given rank, handling multiple devices appropriately
inline void set_device_for_rank(int rank, const device_info& info)
{
    if(info.device_count > 0)
    {
        int device_id = rank % info.device_count;
        HIP_CHECK(hipSetDevice(device_id));
    }
}

/// \brief Print device information for the current device
inline void print_device_info(int rank)
{
    hipDeviceProp_t properties;
    int             device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&properties, device_id));
    std::cout << "[Rank " << rank << "] Device assigned: \"" << properties.name
              << "\" (ID: " << device_id << ")" << std::endl;
}

/// \brief Thread-safe printing utility with mutex protection
template<typename T = void>
class safe_printer_impl
{
private:
    static std::mutex print_mutex_;

public:
    template<typename... Args>
    static void print(Args&&... args)
    {
        std::lock_guard<std::mutex> lock(print_mutex_);
        (std::cout << ... << args);
    }

    template<typename... Args>
    static void printf(const char* format, Args... args)
    {
        std::lock_guard<std::mutex> lock(print_mutex_);
        std::printf(format, args...);
    }
};

template<typename T>
std::mutex safe_printer_impl<T>::print_mutex_{};

using safe_printer = safe_printer_impl<>;

/// \brief Calculate memory bandwidth in GB/sec
inline double calculate_bandwidth_gb_per_sec(size_t bytes, double time_seconds)
{
    if(time_seconds <= 0.0)
        return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / time_seconds;
}

#if defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM) \
    && ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM > 0
namespace fs = ::ghc::filesystem; // NOLINT(misc-unused-alias-decls)
#elif defined(ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM) \
    && ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM > 0
namespace fs = ::std::filesystem; // NOLINT(misc-unused-alias-decls)
#else
namespace fs = ::std::experimental::filesystem; // NOLINT(misc-unused-alias-decls)
#endif

struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};
};

using call_stack_t = std::vector<source_location>;

inline void print_call_stack(std::string         ofname,
                             const call_stack_t& _call_stack,
                             const char*         env_variable = "ROCPROFILER_SAMPLE_OUTPUT_FILE")
{
    if(auto* eofname = getenv(env_variable))
        ofname = eofname;

    std::ostream* ofs     = nullptr;
    auto          cleanup = std::function<void(std::ostream*&)>{};

    if(ofname == "stdout")
        ofs = &std::cout;
    else if(ofname == "stderr")
        ofs = &std::cerr;
    else
    {
        ofs = new std::ofstream{ofname};
        if(ofs && *ofs)
            cleanup = [](std::ostream*& _os) { delete _os; };
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n";
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    std::cout << "Outputting collected data to " << ofname << "...\n" << std::flush;

    size_t n = 0;
    for(const auto& itr : _call_stack)
    {
        *ofs << std::left << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size()
             << " [" << common::fs::path{itr.file}.filename() << ":" << itr.line << "] "
             << std::setw(20) << itr.function;
        if(!itr.context.empty())
            *ofs << " :: " << itr.context;
        *ofs << "\n";
    }

    *ofs << std::flush;

    if(cleanup)
        cleanup(ofs);
}

template<typename Tp>
std::string as_hex(Tp _v, size_t _width = 16)
{
    uintptr_t _vp = 0;
    if constexpr(std::is_pointer<Tp>::value)
        _vp = reinterpret_cast<uintptr_t>(_v);
    else
        _vp = _v;

    auto _ss = std::stringstream{};
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _vp;
    return _ss.str();
}

using callback_name_info = rocprofiler::sdk::callback_name_info;
using buffer_name_info   = rocprofiler::sdk::buffer_name_info;

inline auto get_buffer_tracing_names()
{
    return rocprofiler::sdk::get_buffer_tracing_names();
}

inline auto get_callback_tracing_names()
{
    return rocprofiler::sdk::get_callback_tracing_names();
}

inline std::ostream*& get_output_stream()
{
    // The output strea is initially unitialized
    static std::ostream* _v = nullptr;
    return _v;
}

} // namespace common

#endif // COMMON_ROCPROFILER_UTILS_HPP
