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

#include "backprojection.hpp"
#include "filtering.hpp"
#include "normalization.hpp"
#include "shifting.hpp"
#include "projection.hpp"
#include "utility.hpp"
#include "weighting.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef DATA_DIRECTORY
#define DATA_DIRECTORY "hip_graph_api_tutorial_data"
#endif

auto main() -> int
{
    try
    {
        // Fetch device properties
        auto dev_props = hipDeviceProp_t{};
        hip_check(hipGetDeviceProperties(&dev_props, 0));

        auto const num_streams = dev_props.concurrentKernels;

        std::cout << "Device supports " << num_streams
                  << " concurrent kernels, preprocessing on "
                  << num_streams << " streams." << std::endl;

        auto streams = std::vector<hipStream_t>{};
        streams.resize(num_streams);
        for(auto&& stream : streams)
            hip_check(hipStreamCreate(&stream));

        // Detector constants - all units in mm unless stated otherwise
        constexpr auto d_sd = 553.74f; // Distance between source and detector
        constexpr auto d_so = 210.66f; // Distance between source and origin (object center).
        constexpr auto d_h = 0.05f; // horizontal physical length of a pixel
        constexpr auto d_v = 0.05f; // vertical physical length of a pixel
        constexpr auto theta_step = 0.5f; // angle step size [°]
        constexpr auto shift_h = 4; // horizontal shift [px]
        constexpr auto shift_v = 0; // vertical shift [px]
        auto const delta_h = shift_h * d_h; // physical horizontal shift
        auto const delta_v = shift_v * d_v; // physical vertical shift
        constexpr auto bps = 12; // bits per sample (pixel) in input projection [no unit]

        // Initialized after first projection is loaded - all units are px unless stated otherwise
        auto N_h = std::uint32_t{}; // horizontal size
        auto N_v = std::uint32_t{}; // vertical size
        auto N_hFFT = std::uint32_t{}; // filter length
        auto s_N_hFFT = std::int32_t{}; // signed filter length
        auto N_hTrans = std::uint32_t{}; // transformed length
        auto s_N_hTrans = std::int32_t{}; // signed transformed length
        auto h_min = float{}; // Horizontal physical center of the detector [mm]
        auto v_min = float{}; // Vertical physical center of the detector [mm]

        // Volume constants - initialized after first projection is loaded
        auto alpha = float{};
        auto radius = float{};
        auto d_x = float{}; // physical voxel size in x direction [mm]
        auto d_y = float{}; // physical voxel size in y direction [mm]
        auto d_z = float{}; // physical voxel size in z direction [mm]
        auto N_x = std::size_t{}; // number of voxels in x direction
        auto N_y = std::size_t{}; // number of voxels in y direction
        auto N_z = std::size_t{}; // number of voxels in z direction
        auto num_subvols = int{}; // number of subvolumes
        auto sub_N_z  = std::vector<std::size_t>{}; // number of voxels in z direction for each subvolume

        auto r = static_cast<float*>(nullptr);
        auto R = static_cast<hipfftComplex*>(nullptr);

        auto forward_plans = std::vector<hipfftHandle>{};
        auto forward_sizes = std::vector<std::size_t>{};
        auto backward_plans = std::vector<hipfftHandle>{};
        auto backward_sizes = std::vector<std::size_t>{};
        forward_plans.resize(num_streams);
        forward_sizes.resize(num_streams);
        backward_plans.resize(num_streams);
        backward_sizes.resize(num_streams);

        auto current_angle = 0.f;
        auto host_inputs = std::vector<std::uint16_t*>{};
        auto input_projections = std::vector<std::uint16_t*>{};
        auto input_pitches = std::vector<std::size_t>{};
        auto shifted_projections = std::vector<std::uint16_t*>{};
        auto shifted_pitches = std::vector<std::size_t>{};
        auto projections = std::vector<float*>{};
        auto projection_pitches = std::vector<std::size_t>{};
        auto expanded_projections = std::vector<float*>{};
        auto expanded_pitches = std::vector<std::size_t>{};
        auto transformed_projections = std::vector<hipfftComplex*>{};
        auto transformed_pitches = std::vector<std::size_t>{};
        auto texture_projections = std::vector<hipTextureObject_t>{};

        auto preprocessing_events = std::vector<hipEvent_t>{};
        auto backprojection_events = std::vector<hipEvent_t>{};
        
        auto save_stream = hipStream_t{};
        auto save_event = hipEvent_t{};

        auto const data_path = std::filesystem::path{DATA_DIRECTORY};
        auto data_it = std::filesystem::directory_iterator{data_path};

        // A directory iterator is not guaranteed to go through the elements alphabetically (or another way of sorting).
        // Using a set for the paths guarantees alphabetical ordering.
        auto paths = std::set<std::filesystem::path>{};
        for(auto& entry : data_it)
            paths.insert(entry.path());

        // Load first projection and initialize the invariants
        for(auto const& path : paths)
        {
            if(path.extension() != std::filesystem::path({".tif"}))
                continue;
            else
            {
                std::cout << "Initializing... ";
                get_projection_dims(path.string(), N_h, N_v);

                auto stream = streams.at(0);

                // N_h: number of detector pixels (horizontal)
                // d_h: detector pixel size (mm)
                // delta_h: physical shift (mm), positive values shift detector to the left
                h_min = -((N_h - 1) * d_h) / 2.f + delta_h;
                v_min = -((N_v - 1) * d_v) / 2.f + delta_v;

                //h_min = delta_h - ((d_h * N_h) / 2.f);

                // Filter length
                N_hFFT = /*2u **/ std::pow(2.f, std::ceil(std::log2(N_h)));
                s_N_hFFT = static_cast<std::int32_t>(N_hFFT);

                // Transformed filter length
                N_hTrans = N_hFFT / 2 + 1;
                s_N_hTrans = static_cast<std::int32_t>(N_hTrans);

                // Create filter kernel
                hip_check(hipMalloc(&r, N_hFFT * sizeof(float)));
                auto const creation_blocks = std::max((N_hFFT / 1024), 1u);
                filter_creation_kernel<<<creation_blocks, 1024, 0, stream>>>(r, s_N_hFFT, d_h);

                hip_check(hipMalloc(&R, N_hTrans * sizeof(hipfftComplex)));
                auto filter_plan = hipfftHandle{};
                hipfft_check(hipfftPlan1d(&filter_plan, N_hFFT, HIPFFT_R2C, 1));
                hipfft_check(hipfftSetStream(filter_plan, stream));
                hipfft_check(hipfftExecR2C(filter_plan, r, R));

                auto absolute_blocks = std::max(1u, ((N_hTrans / 1024u) + 1u));
                filter_absolute_kernel<<<absolute_blocks, 1024, 0, stream>>>(R, N_hTrans, d_h);

                hip_check(hipStreamSynchronize(stream));
                
                hipfft_check(hipfftDestroy(filter_plan));
                hip_check(hipFree(r));

                /* Calculate volume dimensions */
                alpha = std::atan((((N_h * d_h) / 2.f) + std::abs(delta_h)) / d_sd);
                radius = std::abs(d_so) * std::sin(alpha);

                d_x = radius / ((((N_h * d_h) / 2.f) + std::abs(delta_h)) / d_h);
                d_y = d_x;
                d_z = d_x;

                N_x = (2.f * radius) / d_x;
                N_y = N_x;
                N_z = ((N_v * d_v / 2.f) + std::abs(delta_v)) * (std::abs(d_so) / d_sd) * (2.f / d_z);

                /* Estimate number of subvolumes. We will split the volume along the z axis */
                auto const input_proj_single = N_h * N_v * sizeof(std::uint16_t);
                auto const input_proj_total = input_proj_single * 2u * num_streams;
                auto const proj_single = N_h * N_v * sizeof(float);
                auto const proj_total = proj_single * num_streams;
                auto const expanded_single = N_hFFT * N_v * sizeof(float);
                auto const expanded_total = expanded_single * num_streams;
                auto const transformed_single = N_hTrans * N_v * sizeof(hipfftComplex);
                auto const transformed_total = transformed_single * num_streams;
                auto const volume_total = N_x * N_y * N_z * sizeof(float);
                auto const mem_total = input_proj_total + proj_total + expanded_total + transformed_total + volume_total;
                auto dev_mem_free = std::size_t{};
                auto dev_mem_total = std::size_t{};
                hip_check(hipMemGetInfo(&dev_mem_free, &dev_mem_total));

                auto mem_required = mem_total;
                num_subvols = 1;
                while(std::max(mem_required, dev_mem_free) == mem_required)
                {
                    mem_required /= 2;
                    num_subvols *= 2;
                }

                // if there is less than a GiB of free memory left on the device, use smaller subvolumes to be safe
                if((dev_mem_free - mem_required) < (1024 * 1024 * 1024))
                    num_subvols *= 2;

                auto const s_N_z = N_z / num_subvols;
                for(auto i = 0; i < num_subvols - 1; ++i)
                    sub_N_z.push_back(s_N_z);

                if(auto const remainder = N_z % num_subvols; remainder == 0)
                    sub_N_z.push_back(s_N_z);
                else
                    sub_N_z.push_back(s_N_z + remainder);

                hip_check(hipStreamCreate(&save_stream));
                hip_check(hipEventCreate(&save_event));

                std::cout << "Done!" << std::endl;
                std::cout << "Volume dimensions: " << N_x << " x " << N_y << " x " << N_z << std::endl;
                if(num_subvols > 1)
                {
                    std::cout << "Number of subvolumes: " << num_subvols << std::endl;
                    for(auto s_N_z : sub_N_z)
                        std::cout << "\tDimensions: " << N_x << " x " << N_y << " x " << s_N_z << std::endl;
                }

                break;
            }
        }

        // Initialize stream data
        auto host_output = static_cast<float*>(nullptr);        
        for(auto stream_idx = 0; stream_idx < streams.size(); ++stream_idx)
        {
            std::cout << "Initializing stream " << stream_idx << "... ";
            auto stream = streams.at(stream_idx);

            auto host_input = static_cast<std::uint16_t*>(nullptr);
            hip_check(hipHostMalloc(&host_input, N_h * N_v * sizeof(std::uint16_t), hipHostMallocWriteCombined));
            host_inputs.push_back(host_input);

            hip_check(hipHostMalloc(&host_output, N_h * N_v * sizeof(float), hipHostMallocDefault));

            auto input = static_cast<std::uint16_t*>(nullptr);
            auto input_pitch = std::size_t{};
            hip_check(hipMallocPitch(&input, &input_pitch, N_h * sizeof(std::uint16_t), N_v));
            input_projections.push_back(input);
            input_pitches.push_back(input_pitch);

            auto shifted = static_cast<std::uint16_t*>(nullptr);
            auto shifted_pitch = std::size_t{};
            hip_check(hipMallocPitch(&shifted, &shifted_pitch, N_h * sizeof(std::uint16_t), N_v));
            shifted_projections.push_back(shifted);
            shifted_pitches.push_back(shifted_pitch);

            auto proj = static_cast<float*>(nullptr);
            auto proj_pitch = std::size_t{};
            hip_check(hipMallocPitch(&proj, &proj_pitch, N_h * sizeof(float), N_v));
            projections.push_back(proj);
            projection_pitches.push_back(proj_pitch);

            auto expanded = static_cast<float*>(nullptr);
            auto expanded_pitch = std::size_t{};
            hip_check(hipMallocPitch(&expanded, &expanded_pitch, N_hFFT * sizeof(float), N_v));
            expanded_projections.push_back(expanded);
            expanded_pitches.push_back(expanded_pitch);
            
            auto transformed = static_cast<hipfftComplex*>(nullptr);
            auto transformed_pitch = std::size_t{};
            hip_check(hipMallocPitch(&transformed, &transformed_pitch, N_hTrans * sizeof(hipfftComplex), N_v));
            transformed_projections.push_back(transformed);
            transformed_pitches.push_back(transformed_pitch);

            auto& forward = forward_plans.at(stream_idx);
            auto& forward_size = forward_sizes.at(stream_idx);
            auto fw_inembed = static_cast<int>(expanded_pitch / sizeof(float));
            auto fw_istride = 1;
            auto fw_idist = fw_inembed;
            auto fw_onembed = static_cast<int>(transformed_pitch / sizeof(hipfftComplex));
            auto fw_ostride = 1;
            auto fw_odist = fw_onembed;
            hipfft_check(hipfftCreate(&forward));
            hipfft_check(hipfftMakePlanMany(forward, 1, &s_N_hFFT,
                                            &fw_inembed, 1, fw_idist,
                                            &fw_onembed, 1, fw_odist,
                                            HIPFFT_R2C, static_cast<int>(N_v), &forward_size));
            hipfft_check(hipfftSetStream(forward, stream));

            auto& backward = backward_plans.at(stream_idx);
            auto& backward_size = backward_sizes.at(stream_idx);
            auto bw_inembed = fw_onembed;
            auto bw_istride = fw_ostride;
            auto bw_idist = fw_odist;
            auto bw_onembed = fw_inembed;
            auto bw_ostride = fw_istride;
            auto bw_odist = fw_idist;
            hipfft_check(hipfftCreate(&backward));
            hipfft_check(hipfftMakePlanMany(backward, 1, &s_N_hFFT,
                                            &bw_inembed, bw_istride, bw_idist,
                                            &bw_onembed, bw_ostride, bw_odist,
                                            HIPFFT_C2R, static_cast<int>(N_v), &backward_size));
            hipfft_check(hipfftSetStream(backward, stream));

            // create a HIP texture from the projection
            auto res_desc = hipResourceDesc{};
            res_desc.resType = hipResourceTypePitch2D;
            res_desc.res.pitch2D.desc = hipCreateChannelDesc<float>();
            res_desc.res.pitch2D.devPtr = static_cast<void*>(proj);
            res_desc.res.pitch2D.width = N_h;
            res_desc.res.pitch2D.height = N_v;
            res_desc.res.pitch2D.pitchInBytes = proj_pitch;

            auto tex_desc = hipTextureDesc{};
            tex_desc.addressMode[0] = hipAddressModeBorder;
            tex_desc.addressMode[1] = hipAddressModeBorder;
            tex_desc.readMode = hipReadModeElementType;
            tex_desc.borderColor[0] = 0.f;
            tex_desc.borderColor[0] = 0.f;
            tex_desc.filterMode = hipFilterModeLinear;
            tex_desc.normalizedCoords = 0;

            auto& p_tex = texture_projections.emplace_back();
            hip_check(hipCreateTextureObject(&p_tex, &res_desc, &tex_desc, nullptr));

            if(stream_idx > 0)
            {
                auto pp_event = hipEvent_t{};
                hip_check(hipEventCreate(&pp_event));
                preprocessing_events.push_back(pp_event);

                auto bp_event = hipEvent_t{};
                hip_check(hipEventCreate(&bp_event));
                backprojection_events.push_back(bp_event);
            }

            std::cout << "Done!" << std::endl;
        }

        auto const s_N_z_first = sub_N_z.at(0); // might be different from s_N_z for the last subvolume
        create_volume("volume.tif");
        auto host_vol_raw_ptr = static_cast<float*>(nullptr);
        hip_check(hipHostMalloc(&host_vol_raw_ptr, N_x * N_y * s_N_z_first * sizeof(float), hipHostMallocDefault));
        auto host_vol_ptr = make_hipPitchedPtr(host_vol_raw_ptr, N_x * sizeof(float), N_x, N_y);

        auto vol_ptr = hipPitchedPtr{};
        auto vol_ext = make_hipExtent(N_x * sizeof(float), N_y, s_N_z_first);
        hip_check(hipMalloc3D(&vol_ptr, vol_ext));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MAIN LOOP
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(auto sub_idx = 3; sub_idx < 4 /* num_subvols */; ++sub_idx)
        {
            auto const s_N_z = sub_N_z.at(sub_idx);
            if(s_N_z != s_N_z_first) // if there is a size mismatch for the last subvolume, delete old one and realloc
            {
                hip_check(hipStreamSynchronize(save_stream)); // Wait for preceding operations on volume to complete
                hip_check(hipFree(vol_ptr.ptr));
                hip_check(hipFreeHost(host_vol_raw_ptr));
                
                hip_check(hipHostMalloc(&host_vol_raw_ptr, N_x * N_y * s_N_z * sizeof(float), hipHostMallocDefault));
                host_vol_ptr = make_hipPitchedPtr(host_vol_raw_ptr, N_x * sizeof(float), N_x, N_y);

                vol_ext = make_hipExtent(N_x * sizeof(float), N_y, s_N_z);
                hip_check(hipMalloc3D(&vol_ptr, vol_ext));
            }

            auto sub_idx_info = new int;
            *sub_idx_info = sub_idx;
            hip_check(hipLaunchHostFunc(streams.at(0), [](void* arg)
            {
                auto my_sub_idx = static_cast<int*>(arg);
                std::cout << "Processing subvolume #" << *my_sub_idx << std::endl;
                delete my_sub_idx;
            }, sub_idx_info));
            hip_check(hipMemset3DAsync(vol_ptr, 0, vol_ext, streams.at(0)));

            auto proj_idx = 0;
            auto stream_idx = 0;
            for(auto const& path : paths)
            {
                if(path.extension() != std::filesystem::path{".tif"})
                    continue;

                // Distribute projections evenly among streams.
                auto stream = streams.at(stream_idx);

                // Using a pointer here ensures that the arguments live beyond the scope of a loop iteration. The
                // pointer will be deleted inside load_projection().
                auto load_proj_args = new load_projection_args
                {
                    path.string(),
                    N_h,
                    N_v,
                    host_inputs.at(stream_idx)
                };
                hip_check(hipLaunchHostFunc(stream, load_projection, static_cast<void*>(load_proj_args)));

                auto host_input = host_inputs.at(stream_idx);
                auto input = input_projections.at(stream_idx);
                auto input_pitch = input_pitches.at(stream_idx);
                hip_check(hipMemcpy2DAsync(input, input_pitch,
                                        host_input, N_h * sizeof(std::uint16_t),
                                        N_h * sizeof(std::uint16_t), N_v,
                                        hipMemcpyHostToDevice, stream));

                auto threads_per_block = dim3{32, 32, 1};
                auto blocks_per_grid = dim3{(N_h / threads_per_block.x) + 1,
                                            (N_v / threads_per_block.y) + 1,
                                            1};

                /*auto shifted = shifted_projections.at(stream_idx);
                auto shifted_pitch = shifted_pitches.at(stream_idx);
                shifting_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(input, input_pitch,
                                                                                  shifted, shifted_pitch,
                                                                                  N_h, N_v,
                                                                                  shift_h, shift_v);*/

                auto proj = projections.at(stream_idx);
                auto proj_pitch = projection_pitches.at(stream_idx);
                normalization_kernel<bps><<<blocks_per_grid, threads_per_block, 0, stream>>>(input, input_pitch,
                                                                                             proj, proj_pitch,
                                                                                             N_h, N_v);

                /*if(sub_idx == 3 && proj_idx == 0)
                {
                    hip_check(hipMemcpy2DAsync(host_output, N_h * sizeof(float),
                                               proj, proj_pitch,
                                               N_h * sizeof(float), N_v,
                                               hipMemcpyDeviceToHost, stream));
                    auto save_args = new save_projection_args
                    {
                        "normalized.tif",
                        N_h, N_v,
                        host_output
                    };
                    hip_check(hipLaunchHostFunc(stream, save_projection<float>, static_cast<void*>(save_args)));
                }*/

                /*weighting_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(proj, proj_pitch, N_h, N_v, d_sd,
                                                                                    h_min, v_min, d_h, d_v);*/

                /*if(sub_idx == 3 && proj_idx == 0)
                {
                    hip_check(hipMemcpy2DAsync(host_output, N_h * sizeof(float),
                                               proj, proj_pitch,
                                               N_h * sizeof(float), N_v,
                                               hipMemcpyDeviceToHost, stream));
                    auto save_args = new save_projection_args
                    {
                        "weighted.tif",
                        N_h, N_v,
                        host_output
                    };
                    hip_check(hipLaunchHostFunc(stream, save_projection<float>, static_cast<void*>(save_args)));
                }*/

                // Expand projection to filter length
                auto expanded = expanded_projections.at(stream_idx);
                auto expanded_pitch = expanded_pitches.at(stream_idx);
                hip_check(hipMemset2DAsync(expanded, expanded_pitch, 0, N_hFFT * sizeof(float), N_v, stream));
                hip_check(hipMemcpy2DAsync(expanded, expanded_pitch,
                                           proj, proj_pitch,
                                           N_h * sizeof(float), N_v, hipMemcpyDeviceToDevice, stream));

                // Fourier-transform projection and filter kernel
                auto transformed = transformed_projections.at(stream_idx);
                auto transformed_pitch = transformed_pitches.at(stream_idx);
                hip_check(hipMemset2DAsync(transformed, transformed_pitch, 0, N_hTrans * sizeof(hipfftComplex), N_v, stream));
                auto& forward = forward_plans.at(stream_idx);
                hipfft_check(hipfftExecR2C(forward, expanded, transformed));
                
                // Apply filter
                auto filter_blocks_per_grid = dim3{(N_hTrans / threads_per_block.x) + 1,
                                                (N_v / threads_per_block.y) + 1,
                                                1};
                filter_application_kernel<<<filter_blocks_per_grid, threads_per_block, 0, stream>>>(transformed,
                                                                                                    transformed_pitch,
                                                                                                    R, N_hTrans, N_v);
                
                auto& backward = backward_plans.at(stream_idx);
                hipfft_check(hipfftExecC2R(backward, transformed, expanded));

                hip_check(hipMemcpy2DAsync(proj, proj_pitch,
                                        expanded, expanded_pitch,
                                        N_h * sizeof(float), N_v,
                                        hipMemcpyDeviceToDevice, stream));
                
                /*if(sub_idx == 3 && proj_idx == 0)
                {
                    hip_check(hipMemcpy2DAsync(host_output, N_h * sizeof(float),
                                               proj, proj_pitch,
                                               N_h * sizeof(float), N_v,
                                               hipMemcpyDeviceToHost, stream));
                    auto save_args = new save_projection_args
                    {
                        "unnormalized.tif",
                        N_h, N_v,
                        host_output
                    };
                    hip_check(hipLaunchHostFunc(stream, save_projection<float>, static_cast<void*>(save_args)));
                }*/

                filter_normalization_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(proj, proj_pitch, N_hFFT,
                                                                                            N_h, N_v);
                
                /*if(sub_idx == 3 && proj_idx == 0)
                {
                    hip_check(hipMemcpy2DAsync(host_output, N_h * sizeof(float),
                                               proj, proj_pitch,
                                               N_h * sizeof(float), N_v,
                                               hipMemcpyDeviceToHost, stream));
                    auto save_args = new save_projection_args
                    {
                        "filtered.tif",
                        N_h, N_v,
                        host_output
                    };
                    hip_check(hipLaunchHostFunc(stream, save_projection<float>, static_cast<void*>(save_args)));
                    hip_check(hipStreamSynchronize(stream));
                    std::exit(EXIT_SUCCESS);
                }*/

                // The backprojection always runs in the first stream. If multiple streams are preprocessing, let the
                // first stream wait for the current projection to finish preprocessing.
                if(stream_idx > 0)
                {
                    auto pp_event = preprocessing_events.at(stream_idx - 1);
                    hip_check(hipEventRecord(pp_event, stream));
                    hip_check(hipStreamWaitEvent(streams.at(0), pp_event));
                }

                // Backprojection
                auto const theta_deg = theta_step * proj_idx; // Angle of current projection in degrees
                auto const theta_rad = theta_deg * std::numbers::pi_v<float> / 180.f; // Convert to radians
                auto const sin_theta = std::sin(theta_rad);
                auto const cos_theta = std::cos(theta_rad);

                auto bp_block_size = dim3{32u, 8u, 4u};
                auto bp_blocks = dim3{static_cast<std::uint32_t>(N_x / bp_block_size.x + 1),
                                      static_cast<std::uint32_t>(N_y / bp_block_size.y + 1),
                                      static_cast<std::uint32_t>(s_N_z / bp_block_size.z + 1)};

                auto& p_tex = texture_projections.at(stream_idx);
                backprojection_kernel<<<bp_blocks, bp_block_size, 0, streams.at(0)>>>(static_cast<float*>(vol_ptr.ptr),
                                                                                      vol_ptr.pitch,
                                                                                      N_x, N_y, N_z,
                                                                                      s_N_z, s_N_z_first * sub_idx,
                                                                                      d_x, d_y, d_z,
                                                                                      p_tex, h_min, v_min,
                                                                                      sin_theta, cos_theta, d_h, d_v,
                                                                                      d_sd, d_so);
                hip_check(hipStreamSynchronize(streams.at(0)));

                // The backprojection always runs in the first stream. If multiple streams are preprocessing, let the
                // current stream wait for the backprojection to finish before reusing the projection memory.
                if(stream_idx > 0)
                {
                    auto bp_event = backprojection_events.at(stream_idx - 1);
                    hip_check(hipEventRecord(bp_event, streams.at(0)));
                    hip_check(hipStreamWaitEvent(stream, bp_event));
                }

                stream_idx = (stream_idx + 1) % streams.size();
                proj_idx++;
            }
            
            auto memcpy_params = hipMemcpy3DParms{};
            std::memset(&memcpy_params, 0, sizeof(hipMemcpy3DParms));
            memcpy_params.dstPos = make_hipPos(0, 0, 0);
            memcpy_params.dstPtr = host_vol_ptr;
            memcpy_params.srcPos = make_hipPos(0, 0, 0);
            memcpy_params.srcPtr = vol_ptr;
            memcpy_params.extent = vol_ext;
            memcpy_params.kind = hipMemcpyDeviceToHost;
            hip_check(hipMemcpy3DAsync(&memcpy_params, streams.at(0)));
            hip_check(hipEventRecord(save_event, streams.at(0)));
            
            auto save_vol_args = new save_volume_args
            {
                "volume.tif",
                host_vol_raw_ptr,
                N_x, N_y, N_z, s_N_z,
                d_x, d_y
            };
            hip_check(hipStreamWaitEvent(save_stream, save_event));
            hip_check(hipLaunchHostFunc(save_stream, save_volume, save_vol_args));
        }

        std::cout << "All work items enqueued, waiting for completion...";
        hip_check(hipStreamSynchronize(save_stream));
        std::cout << "Done!" << std::endl;

        hip_check(hipFree(vol_ptr.ptr));
        hip_check(hipFreeHost(host_vol_raw_ptr));

        for(auto&& e : backprojection_events)
            hip_check(hipEventDestroy(e));

        for(auto&& e : preprocessing_events)
            hip_check(hipEventDestroy(e));

        for(auto&& tex : texture_projections)
            hip_check(hipDestroyTextureObject(tex));

        for(auto&& plan : backward_plans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& plan : forward_plans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& p : transformed_projections)
            hip_check(hipFree(p));

        for(auto&& p : expanded_projections)
            hip_check(hipFree(p));

        for(auto&& p : projections)
            hip_check(hipFree(p));

        for(auto&& p : shifted_projections)
            hip_check(hipFree(p));

        for(auto&& p : input_projections)
            hip_check(hipFree(p));

        for(auto&& p : host_inputs)
            hip_check(hipFreeHost(p));

        hip_check(hipEventDestroy(save_event));
        hip_check(hipStreamDestroy(save_stream));

        hip_check(hipFree(R));

        for(auto&& stream : streams)
            hip_check(hipStreamDestroy(stream));

        return EXIT_SUCCESS;
    }
    catch(std::filesystem::filesystem_error const& e)
    {
        std::cerr << "Caught filesystem error: " << e.what() << " (" << e.code() << ')' << std::endl;
        return EXIT_FAILURE;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
