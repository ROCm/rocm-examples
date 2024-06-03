// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <iostream>
#include <vector>

/// \brief Computes the histogram of the texture object.
__global__ void histogram_kernel(unsigned int*      histogram,
                                 unsigned int       size_x,
                                 unsigned int       size_y,
                                 unsigned int       hist_bin_count,
                                 hipTextureObject_t tex_obj)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= size_x || y >= size_y)
    {
        return;
    }

    // Normalize coordinates, add 0.5 to sample the middle of the texel.
    float u = x / static_cast<float>(size_x) + .5f;
    float v = y / static_cast<float>(size_y) + .5f;

#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    // Read the value from the texture, if supported.
    unsigned char val = tex2D<unsigned char>(tex_obj, u, v);
#else
    // Prevent compile errors on HIP architectures that do not support
    // texture instructions.
    (void)u;
    (void)v;
    (void)tex_obj;
    unsigned char val = 0;
#endif

    // Determine the histogram bin and write to global memory.
    unsigned int bin_range = ceiling_div(256, hist_bin_count);
    unsigned int bin_idx   = static_cast<unsigned int>(val) / bin_range;
    atomicAdd(&histogram[bin_idx], 1);
}

/// \brief Returns \p true if texture functions are supported for device <tt>device_id</tt>, \p false otherwise.
static bool is_image_supported(int device_id)
{
#ifdef __HIP_PLATFORM_AMD__
    int image_support;
    HIP_CHECK(hipDeviceGetAttribute(&image_support, hipDeviceAttributeImageSupport, device_id));
    return image_support == 1;
#else
    (void)device_id;
    // hipDeviceAttributeImageSupport is not supported for the NVIDIA platform.
    #if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    return true;
    #else
    return false;
    #endif
#endif
}

int main()
{
    if(!is_image_supported(0))
    {
        std::cout << "Texture functions are not supported on device 0." << std::endl;
        return 0;
    }

    constexpr unsigned int size_x = 1024;
    constexpr unsigned int size_y = 1024;
    constexpr unsigned int size   = size_x * size_y;

    // Allocate and set host data.
    std::vector<unsigned char> h_data(size);
    for(unsigned int i = 0; i < size; i++)
    {
        h_data[i] = static_cast<unsigned char>(i);
    }

    // Allocate hip array in device memory.
    hipChannelFormatDesc channel_desc
        = hipCreateChannelDesc(sizeof(unsigned char) * 8, 0, 0, 0, hipChannelFormatKindUnsigned);
    hipArray_t d_array;
    HIP_CHECK(hipMallocArray(&d_array, &channel_desc, size_x, size_y));

    // The pitch of the source memory, which is the width in memory in bytes of the 2D array pointed
    // to by h_data, including padding. We don't have any padding.
    const size_t spitch = size_x * sizeof(unsigned char);
    // Copy the data located at address h_data in host memory to device memory.
    HIP_CHECK(hipMemcpy2DToArray(d_array,
                                 0,
                                 0,
                                 h_data.data(),
                                 spitch,
                                 size_x * sizeof(unsigned char),
                                 size_y,
                                 hipMemcpyHostToDevice));

    // Specify the texture resource.
    hipResourceDesc res_desc{};
    res_desc.resType         = hipResourceTypeArray;
    res_desc.res.array.array = d_array;

    // Specify the texture object parameters.
    hipTextureDesc tex_desc{};
    tex_desc.addressMode[0]   = hipAddressModeWrap;
    tex_desc.addressMode[1]   = hipAddressModeWrap;
    tex_desc.filterMode       = hipFilterModePoint;
    tex_desc.readMode         = hipReadModeElementType;
    tex_desc.normalizedCoords = 1;

    // Create the texture object.
    hipTextureObject_t tex_obj{};
    HIP_CHECK(hipCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    constexpr unsigned int hist_bin_count = 7;
    constexpr size_t       hist_bytes     = hist_bin_count * sizeof(unsigned int);

    // Allocate the histogram in device memory.
    unsigned int* d_histogram{};
    HIP_CHECK(hipMalloc(&d_histogram, hist_bytes));
    HIP_CHECK(hipMemset(d_histogram, 0, hist_bytes));

    // Invoke histogram kernel.
    constexpr unsigned int block_dim = 16;
    histogram_kernel<<<dim3(ceiling_div(size_x, block_dim), ceiling_div(size_y, block_dim)),
                       dim3(block_dim, block_dim),
                       0,
                       hipStreamDefault>>>(d_histogram, size_x, size_y, hist_bin_count, tex_obj);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy data from device back to host.
    std::array<unsigned int, hist_bin_count> h_histogram;
    HIP_CHECK(hipMemcpy(h_histogram.data(), d_histogram, hist_bytes, hipMemcpyDeviceToHost));

    // Print out results.
    std::cout << "Equal-width histogram with " << hist_bin_count << " bins of values [0, " << size
              << ") mod 256:\n";
    unsigned int sum = 0;
    for(unsigned int i = 0; i < hist_bin_count; i++)
    {
        std::cout << "bin[" << i << "] = " << h_histogram[i];
        std::cout << (i + 1 < hist_bin_count ? ", " : "\n");
        sum += h_histogram[i];
    }
    std::cout << "sum of bins: " << sum << std::endl;

    // Destroy texture object.
    HIP_CHECK(hipDestroyTextureObject(tex_obj));

    // Free device memory.
    HIP_CHECK(hipFreeArray(d_array));
    HIP_CHECK(hipFree(d_histogram));
}
