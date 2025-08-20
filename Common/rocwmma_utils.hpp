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

#ifndef COMMON_ROCWMMA_UTILS_HPP
#define COMMON_ROCWMMA_UTILS_HPP

#include "example_utils.hpp"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <iostream>

/// \brief Get current device warp size
inline uint32_t get_warp_size()
{
    hipDeviceProp_t device_prop;
    int             device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&device_prop, device_id));
    return device_prop.warpSize;
}

/// \brief Check if current device supports F64 operations
inline bool is_f64_supported()
{
    hipDevice_t     handle;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&handle));
    HIP_CHECK(hipGetDeviceProperties(&props, handle));

    std::string device_name(props.gcnArchName);

    return ((device_name.find("gfx90a") != std::string::npos)
            || (device_name.find("gfx942") != std::string::npos)
            || (device_name.find("gfx950") != std::string::npos));
}

/// \brief Check if current device supports F32 operations
inline bool is_f32_supported()
{
    hipDevice_t     handle;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&handle));
    HIP_CHECK(hipGetDeviceProperties(&props, handle));

    std::string device_name(props.gcnArchName);

    return ((device_name.find("gfx908") != std::string::npos)
            || (device_name.find("gfx90a") != std::string::npos)
            || (device_name.find("gfx942") != std::string::npos)
            || (device_name.find("gfx950") != std::string::npos));
}

/// \brief Calculate GFlops for GEMM operation
inline double calculate_gflops(uint32_t m, uint32_t n, uint32_t k)
{
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
}

/// \brief Calculate TFlops per second
inline double calculate_tflops_per_sec(
    uint32_t m, uint32_t n, uint32_t k, double elapsed_time_ms, uint32_t repeats = 1u)
{
    return calculate_gflops(m, n, k) / elapsed_time_ms * static_cast<double>(repeats);
}

/// \brief Matrix initialization with random values
template<typename data_t>
__host__ static inline void fill_rand(data_t* mat, uint32_t m, uint32_t n)
{
    auto rand_init = []()
    {
        srand(time(0));
        return 0u;
    };
    static auto init = rand_init();
    (void) init;

    for(uint32_t i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(uint32_t j = 0; j < n; j++)
        {
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<data_t>::value)
                                 ? -static_cast<data_t>(value)
                                 : static_cast<data_t>(value);
        }
    }
}

/// \brief CPU GEMM reference implementation
template<typename input_t,
         typename output_t,
         typename compute_t,
         typename layout_a,
         typename layout_b,
         typename layout_c,
         typename layout_d = layout_c>
__host__ void gemm_cpu_h(uint32_t        m,
                         uint32_t        n,
                         uint32_t        k,
                         input_t const*  a,
                         input_t const*  b,
                         output_t const* c,
                         output_t*       d,
                         uint32_t        lda,
                         uint32_t        ldb,
                         uint32_t        ldc,
                         uint32_t        ldd,
                         compute_t       alpha,
                         compute_t       beta)
{
    auto row_mjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto col_mjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto a_index = std::is_same<layout_a, rocwmma::row_major>::value ? row_mjr : col_mjr;
    auto b_index = std::is_same<layout_b, rocwmma::row_major>::value ? row_mjr : col_mjr;
    auto c_index = std::is_same<layout_c, rocwmma::row_major>::value ? row_mjr : col_mjr;
    auto d_index = std::is_same<layout_d, rocwmma::row_major>::value ? row_mjr : col_mjr;

    for(uint32_t i = 0; i < m; ++i)
    {
        for(uint32_t j = 0; j < n; ++j)
        {
            compute_t accum = static_cast<compute_t>(0);
            for(uint32_t h = 0; h < k; ++h)
            {
                accum += static_cast<compute_t>(a[a_index(i, h, lda)])
                         * static_cast<compute_t>(b[b_index(h, j, ldb)]);
            }
            d[d_index(i, j, ldd)] = static_cast<output_t>(
                alpha * accum + beta * static_cast<compute_t>(c[c_index(i, j, ldc)]));
        }
    }
}

/// \brief Element-wise comparison
template<typename data_t>
__host__ std::pair<bool, double>
         compare_equal(data_t const* a, data_t const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    auto to_double = [](data_t const& val) { return static_cast<double>(static_cast<float>(val)); };

    for(uint32_t i = 0; i < size; ++i)
    {
        auto val_a = a[i];
        auto val_b = b[i];

        auto numerator = fabs(to_double(val_a) - to_double(val_b));
        auto divisor   = fabs(to_double(val_a)) + fabs(to_double(val_b)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
            retval             = false;
            max_relative_error = std::numeric_limits<data_t>::infinity();
            break;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
                retval             = false;
                max_relative_error = std::numeric_limits<data_t>::signaling_NaN();
                break;
            }
            else if(relative_error > max_relative_error)
            {
                max_relative_error = relative_error;
            }
        }
    }

    auto eps = to_double(std::numeric_limits<data_t>::epsilon());
    if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

/// \brief Matrix initialization with batch support
template<typename data_t>
__host__ static inline void
    fill(data_t* mat, uint32_t m, uint32_t k, uint32_t b, uint32_t normalization = 1)
{
    auto batch_offset = m * k;
    for(uint32_t t = 0; t < b; ++t)
    {
        for(uint32_t i = 0; i < m; ++i)
        {
            for(uint32_t j = 0; j < k; ++j)
            {
                auto value
                    = static_cast<float>(rand() / normalization) / static_cast<float>(RAND_MAX);
                mat[t * batch_offset + i * k + j] = static_cast<data_t>(value);
            }
        }
    }
}

/// \brief Check if current device is GFX9 architecture
inline bool is_gfx9()
{
    hipDevice_t     handle;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&handle));
    HIP_CHECK(hipGetDeviceProperties(&props, handle));

    std::string device_name(props.gcnArchName);

    return ((device_name.find("gfx908") != std::string::npos)
            || (device_name.find("gfx90a") != std::string::npos)
            || (device_name.find("gfx942") != std::string::npos)
            || (device_name.find("gfx950") != std::string::npos));
}

/// \brief Check if current device is GFX11 architecture
inline bool is_gfx11()
{
    hipDevice_t     handle;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&handle));
    HIP_CHECK(hipGetDeviceProperties(&props, handle));

    std::string device_name(props.gcnArchName);

    return ((device_name.find("gfx1100") != std::string::npos)
            || (device_name.find("gfx1101") != std::string::npos)
            || (device_name.find("gfx1102") != std::string::npos)
            || (device_name.find("gfx1151") != std::string::npos));
}

/// \brief Check if current device is GFX12 architecture
inline bool is_gfx12()
{
    hipDevice_t     handle;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&handle));
    HIP_CHECK(hipGetDeviceProperties(&props, handle));

    std::string device_name(props.gcnArchName);

    return ((device_name.find("gfx1200") != std::string::npos)
            || (device_name.find("gfx1201") != std::string::npos));
}

/// \brief Check HIPRTC error and exit on failure
#ifndef HIPRTC_CHECK
    #define HIPRTC_CHECK(expression)                             \
        if(auto status = (expression); status != HIPRTC_SUCCESS) \
        {                                                        \
            fprintf(stderr,                                      \
                    "hipRTC error: '%s'(%d) at %s:%d\n",         \
                    hiprtcGetErrorString(status),                \
                    status,                                      \
                    __FILE__,                                    \
                    __LINE__);                                   \
            exit(error_exit_code);                               \
        }
#endif

#endif // COMMON_ROCWMMA_UTILS_HPP
