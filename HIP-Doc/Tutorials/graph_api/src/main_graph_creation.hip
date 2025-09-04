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
#include "log_transform.hpp"
#include "normalization.hpp"
#include "projection.hpp"
#include "utility.hpp"
#include "weighting.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <set>
#include <stdexcept>
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

        auto const num_branches = dev_props.concurrentKernels;
        auto const alignment = dev_props.textureAlignment; // This will be used for a custom pitched allocation
        auto get_pitch = [alignment](std::size_t requested)
        {
            if(auto const remainder = requested % alignment; remainder == 0)
                return requested;
            else
                return requested + alignment - remainder;
        };

        // For interoperability with hipFFT we still require streams
        auto streams = std::vector<hipStream_t>{};
        streams.resize(num_branches);
        for(auto&& stream : streams)
            hip_check(hipStreamCreate(&stream));

        std::cout << "Device supports " << num_branches << " concurrent kernels, preprocessing in "
                  << num_branches << " branches." << std::endl;

        auto proj_geom = projection_geometry{};
        auto vol_geom = volume_geometry{};
        auto num_subvols = int{}; // number of subvolumes
        auto sub_N_z  = std::vector<std::size_t>{}; // number of voxels in z direction for each subvolume

        auto r = static_cast<float*>(nullptr);
        auto R = static_cast<hipfftComplex*>(nullptr);

        auto filter_plan = hipfftHandle{};
        auto forward_plans = std::vector<hipfftHandle>{};
        auto forward_sizes = std::vector<std::size_t>{};
        auto backward_plans = std::vector<hipfftHandle>{};
        auto backward_sizes = std::vector<std::size_t>{};
        forward_plans.resize(num_branches);
        forward_sizes.resize(num_branches);
        backward_plans.resize(num_branches);
        backward_sizes.resize(num_branches);

        auto current_angle = 0.f;
        auto host_inputs = std::vector<std::uint16_t*>{};
        auto input_projections = std::vector<std::uint16_t*>{};
        auto input_pitches = std::vector<std::size_t>{};
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
            if(path.extension() != std::filesystem::path{".tif"})
                continue;
            else
            {
                std::cout << "Initializing... ";
                proj_geom = projection_geometry{path.string()};
                vol_geom = volume_geometry{proj_geom};

                auto stream = streams.at(0);

                // Create filter kernel
                hip_check(hipMalloc(&r, proj_geom.N_hFFT * sizeof(float)));
                auto const creation_blocks = std::max((proj_geom.N_hFFT / 1024), 1u);
                filter_creation_kernel<<<creation_blocks, 1024, 0, stream>>>(r, proj_geom.s_N_hFFT, proj_geom.d_h);

                hip_check(hipMalloc(&R, proj_geom.N_hTrans * sizeof(hipfftComplex)));
                auto filter_plan = hipfftHandle{};
                hipfft_check(hipfftPlan1d(&filter_plan, proj_geom.N_hFFT, HIPFFT_R2C, 1));
                hipfft_check(hipfftSetStream(filter_plan, stream));
                hipfft_check(hipfftExecR2C(filter_plan, r, R));

                auto absolute_blocks = (proj_geom.N_hTrans / 1024u) + 1u;
                filter_absolute_kernel<<<absolute_blocks, 1024, 0, stream>>>(R, proj_geom.N_hTrans, proj_geom.d_h);

                hip_check(hipStreamSynchronize(stream));
                
                hipfft_check(hipfftDestroy(filter_plan));
                hip_check(hipFree(r));

                /* Estimate number of subvolumes. We will split the volume along the z axis */
                auto const input_proj_single = proj_geom.N_h * proj_geom.N_v * sizeof(std::uint16_t);
                auto const input_proj_total = input_proj_single * 2u * num_branches;
                auto const proj_single = proj_geom.N_h * proj_geom.N_v * sizeof(float);
                auto const proj_total = proj_single * num_branches;
                auto const expanded_single = proj_geom.N_hFFT * proj_geom.N_v * sizeof(float);
                auto const expanded_total = expanded_single * num_branches;
                auto const transformed_single = proj_geom.N_hTrans * proj_geom.N_v * sizeof(hipfftComplex);
                auto const transformed_total = transformed_single * num_branches;
                auto const volume_total = vol_geom.N_x * vol_geom.N_y * vol_geom.N_z * sizeof(float);
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

                auto const s_N_z = vol_geom.N_z / num_subvols;
                for(auto i = 0; i < num_subvols - 1; ++i)
                    sub_N_z.push_back(s_N_z);

                if(auto const remainder = vol_geom.N_z % num_subvols; remainder == 0)
                    sub_N_z.push_back(s_N_z);
                else
                    sub_N_z.push_back(s_N_z + remainder);

                hip_check(hipStreamCreate(&save_stream));
                hip_check(hipEventCreate(&save_event));

                std::cout << "Done!" << std::endl;
                std::cout << "Volume dimensions: " << vol_geom.N_x << " x "
                                                   << vol_geom.N_y << " x "
                                                   << vol_geom.N_z << std::endl;
                if(num_subvols > 1)
                {
                    std::cout << "Number of subvolumes: " << num_subvols << std::endl;
                    for(auto s_N_z : sub_N_z)
                        std::cout << "\tDimensions: " << vol_geom.N_x << " x "
                                                      << vol_geom.N_y << " x "
                                                      << s_N_z << std::endl;
                }

                break;
            }
        }

        // Initialize per-branch data
        for(auto branch_idx = 0; branch_idx < num_branches; ++branch_idx)
        {
            std::cout << "Initializing branch #" << branch_idx << "...";
            auto stream = streams.at(branch_idx);

            auto host_input = static_cast<std::uint16_t*>(nullptr);
            hip_check(hipHostMalloc(&host_input,
                                    proj_geom.N_h * proj_geom.N_v * sizeof(std::uint16_t),
                                    hipHostMallocWriteCombined));
            host_inputs.push_back(host_input);

            auto input = static_cast<std::uint16_t*>(nullptr);
            auto input_pitch = std::size_t{};
            hip_check(hipMallocPitch(&input, &input_pitch, proj_geom.N_h * sizeof(std::uint16_t), proj_geom.N_v));
            input_projections.push_back(input);
            input_pitches.push_back(input_pitch);

            auto proj = static_cast<float*>(nullptr);
            auto proj_pitch = std::size_t{};
            hip_check(hipMallocPitch(&proj, &proj_pitch, proj_geom.N_h * sizeof(float), proj_geom.N_v));
            projections.push_back(proj);
            projection_pitches.push_back(proj_pitch);

            auto expanded = static_cast<float*>(nullptr);
            auto expanded_pitch = std::size_t{};
            hip_check(hipMallocPitch(&expanded, &expanded_pitch, proj_geom.N_hFFT * sizeof(float), proj_geom.N_v));
            expanded_projections.push_back(expanded);
            expanded_pitches.push_back(expanded_pitch);

            auto transformed = static_cast<hipfftComplex*>(nullptr);
            auto transformed_pitch = std::size_t{};
            hip_check(hipMallocPitch(
                &transformed, &transformed_pitch, proj_geom.N_hTrans * sizeof(hipfftComplex), proj_geom.N_v
            ));
            transformed_projections.push_back(transformed);
            transformed_pitches.push_back(transformed_pitch);

            auto& forward = forward_plans.at(branch_idx);
            auto& forward_size = forward_sizes.at(branch_idx);
            auto fw_inembed = static_cast<int>(expanded_pitch / sizeof(float));
            auto fw_istride = 1;
            auto fw_idist = fw_inembed;
            auto fw_onembed = static_cast<int>(transformed_pitch / sizeof(hipfftComplex));
            auto fw_ostride = 1;
            auto fw_odist = fw_onembed;
            hipfft_check(hipfftCreate(&forward));
            hipfft_check(hipfftMakePlanMany(forward, 1, &proj_geom.s_N_hFFT,
                                            &fw_inembed, 1, fw_idist,
                                            &fw_onembed, 1, fw_odist,
                                            HIPFFT_R2C, static_cast<int>(proj_geom.N_v), &forward_size));
            hipfft_check(hipfftSetStream(forward, stream));

            auto& backward = backward_plans.at(branch_idx);
            auto& backward_size = backward_sizes.at(branch_idx);
            auto bw_inembed = fw_onembed;
            auto bw_istride = fw_ostride;
            auto bw_idist = fw_odist;
            auto bw_onembed = fw_inembed;
            auto bw_ostride = fw_istride;
            auto bw_odist = fw_idist;
            hipfft_check(hipfftCreate(&backward));
            hipfft_check(hipfftMakePlanMany(backward, 1, &proj_geom.s_N_hFFT,
                                            &bw_inembed, bw_istride, bw_idist,
                                            &bw_onembed, bw_ostride, bw_odist,
                                            HIPFFT_C2R, static_cast<int>(proj_geom.N_v), &backward_size));
            hipfft_check(hipfftSetStream(backward, stream));

            // create a HIP texture from the projection
            auto res_desc = hipResourceDesc{};
            res_desc.resType = hipResourceTypePitch2D;
            res_desc.res.pitch2D.desc = hipCreateChannelDesc<float>();
            res_desc.res.pitch2D.devPtr = static_cast<void*>(proj);
            res_desc.res.pitch2D.width = proj_geom.N_h;
            res_desc.res.pitch2D.height = proj_geom.N_v;
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

            std::cout << "Done!" << std::endl;
        }

        auto const s_N_z_first = sub_N_z.front(); // Use the initial subvolume size for offset calculations
        auto const s_N_z_last = sub_N_z.back(); // If there is a largest subvolume, it will be the last
        create_volume("volume.tif");
        auto host_vol_raw_ptr = static_cast<float*>(nullptr);
        hip_check(hipHostMalloc(
            &host_vol_raw_ptr, vol_geom.N_x * vol_geom.N_y * s_N_z_last * sizeof(float), hipHostMallocDefault
        ));
        auto host_vol_ptr = make_hipPitchedPtr(
            host_vol_raw_ptr, vol_geom.N_x * sizeof(float), vol_geom.N_x, vol_geom.N_y
        );    
        auto vol_ptr = hipPitchedPtr{};
        auto vol_ext = make_hipExtent(vol_geom.N_x * sizeof(float), vol_geom.N_y, s_N_z_last);
        hip_check(hipMalloc3D(&vol_ptr, vol_ext));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MAIN LOOP
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto graphCreated = false;
        auto graph = hipGraph_t{};
        auto graphExec = hipGraphExec_t{};
        auto graphStream = hipStream_t{};
        auto graphEvent = hipEvent_t{};
        hip_check(hipStreamCreate(&graphStream));
        hip_check(hipEventCreate(&graphEvent));
        
        for(auto sub_idx = 3; sub_idx < 4 /*num_subvols*/; ++sub_idx)
        {
            auto s_N_z = sub_N_z.at(sub_idx);

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
            auto branch_idx = 0;
            auto preceding_bp_kernel_node = hipGraphNode_t{};
            for(auto const& path : paths)
            {
                if(path.extension() != std::filesystem::path{".tif"})
                    continue;

                if(proj_idx >= proj_geom.num_proj) // We do not want to read both the 0° and 360° projections; skip 360°
                    break;

                // Create a new graph template whenever we reach the first branch again
                if(branch_idx == 0)
                    hip_check(hipGraphCreate(&graph, 0));

                // Distribute projections evenly among branches / streams.
                auto stream = streams.at(branch_idx);

                auto d_sd = proj_geom.d_sd;
                auto d_so = proj_geom.d_so;
                auto h_min = proj_geom.h_min;
                auto v_min = proj_geom.v_min;
                auto d_h = proj_geom.d_h;
                auto d_v = proj_geom.d_v;

                // Using a pointer here ensures that the arguments live beyond the scope of a loop iteration. The
                // pointer will be deleted inside load_projection().
                auto load_proj_args = new load_projection_args
                {
                    path.string(),
                    proj_geom.N_h,
                    proj_geom.N_v,
                    host_inputs.at(branch_idx)
                };
                auto load_proj_node_params = hipHostNodeParams{};
                load_proj_node_params.fn = load_projection;
                load_proj_node_params.userData = static_cast<void*>(load_proj_args);
                auto load_proj_node = hipGraphNode_t{};
                hip_check(hipGraphAddHostNode(&load_proj_node, graph, nullptr, 0, &load_proj_node_params));

                auto host_input = host_inputs.at(branch_idx);
                auto input = input_projections.at(branch_idx);
                auto input_pitch = input_pitches.at(branch_idx);
                auto copy_input_node_params = hipMemcpy3DParms{};
                std::memset(&copy_input_node_params, 0, sizeof(hipMemcpy3DParms));
                copy_input_node_params.srcPos = make_hipPos(0, 0, 0);
                copy_input_node_params.srcPtr = make_hipPitchedPtr(
                    static_cast<void*>(host_input), proj_geom.N_h * sizeof(std::uint16_t), proj_geom.N_h, proj_geom.N_v
                );
                copy_input_node_params.dstPos = make_hipPos(0, 0, 0);
                copy_input_node_params.dstPtr = make_hipPitchedPtr(
                    static_cast<void*>(input), input_pitch, proj_geom.N_h, proj_geom.N_v
                );
                copy_input_node_params.extent = make_hipExtent(
                    proj_geom.N_h * sizeof(std::uint16_t), proj_geom.N_v, 1
                );
                copy_input_node_params.kind = hipMemcpyHostToDevice;
                auto copy_input_node = hipGraphNode_t{};
                hip_check(hipGraphAddMemcpyNode(
                    &copy_input_node, graph, &load_proj_node, 1, &copy_input_node_params
                ));

                auto threads_per_block = dim3{32, 32, 1};
                auto blocks_per_grid = dim3{
                    (proj_geom.N_h / threads_per_block.x) + 1, (proj_geom.N_v / threads_per_block.y) + 1, 1};

                auto proj = projections.at(branch_idx);
                auto proj_pitch = projection_pitches.at(branch_idx);
                void* normalization_kernel_params[] =
                {
                    static_cast<void*>(&input), static_cast<void*>(&input_pitch),
                    static_cast<void*>(&proj), static_cast<void*>(&proj_pitch),
                    static_cast<void*>(&proj_geom.N_h), static_cast<void*>(&proj_geom.N_v)
                };
                auto normalization_kernel_node_params = hipKernelNodeParams{};
                normalization_kernel_node_params.blockDim = threads_per_block;
                normalization_kernel_node_params.extra = nullptr;
                normalization_kernel_node_params.func = reinterpret_cast<void*>(normalization_kernel<proj_geom.bps>);
                normalization_kernel_node_params.gridDim = blocks_per_grid;
                normalization_kernel_node_params.kernelParams = normalization_kernel_params;
                normalization_kernel_node_params.sharedMemBytes = 0;
                auto normalization_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &normalization_kernel_node, graph, &copy_input_node, 1, &normalization_kernel_node_params
                ));

                void* log_transformation_kernel_params[] =
                {
                    static_cast<void*>(&proj), static_cast<void*>(&proj_pitch),
                    static_cast<void*>(&proj_geom.N_h), static_cast<void*>(&proj_geom.N_v)
                };
                auto log_transformation_kernel_node_params = hipKernelNodeParams{};
                log_transformation_kernel_node_params.blockDim = threads_per_block;
                log_transformation_kernel_node_params.extra = nullptr;
                log_transformation_kernel_node_params.func = reinterpret_cast<void*>(log_transformation_kernel);
                log_transformation_kernel_node_params.gridDim = blocks_per_grid;
                log_transformation_kernel_node_params.kernelParams = log_transformation_kernel_params;
                log_transformation_kernel_node_params.sharedMemBytes = 0;
                auto log_transformation_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &log_transformation_kernel_node,
                    graph,
                    &normalization_kernel_node,
                    1,
                    &log_transformation_kernel_node_params
                ));

                void* weighting_kernel_params[] =
                {
                    static_cast<void*>(&proj), static_cast<void*>(&proj_pitch),
                    static_cast<void*>(&proj_geom.N_h), static_cast<void*>(&proj_geom.N_v),
                    static_cast<void*>(&d_sd), static_cast<void*>(&d_so),
                    static_cast<void*>(&h_min), static_cast<void*>(&v_min),
                    static_cast<void*>(&d_h), static_cast<void*>(&d_v)
                };
                auto weighting_kernel_node_params = hipKernelNodeParams{};
                weighting_kernel_node_params.blockDim = threads_per_block;
                weighting_kernel_node_params.extra = nullptr;
                weighting_kernel_node_params.func = reinterpret_cast<void*>(weighting_kernel);
                weighting_kernel_node_params.gridDim = blocks_per_grid;
                weighting_kernel_node_params.kernelParams = weighting_kernel_params;
                weighting_kernel_node_params.sharedMemBytes = 0;
                auto weighting_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &weighting_kernel_node, graph, &log_transformation_kernel_node, 1, &weighting_kernel_node_params
                ));

                // Expand projection to filter length
                auto expanded = expanded_projections.at(branch_idx);
                auto expanded_pitch = expanded_pitches.at(branch_idx);
                auto expanded_memset_node_params = hipMemsetParams{};
                expanded_memset_node_params.dst = static_cast<void*>(expanded);
                expanded_memset_node_params.elementSize = sizeof(float);
                expanded_memset_node_params.height = proj_geom.N_v;
                expanded_memset_node_params.pitch = expanded_pitch;
                expanded_memset_node_params.value = 0;
                expanded_memset_node_params.width = proj_geom.N_h;
                auto expanded_memset_node = hipGraphNode_t{};
                hip_check(hipGraphAddMemsetNode(
                    &expanded_memset_node, graph, nullptr, 0, &expanded_memset_node_params
                ));

                auto copy_proj_to_expanded_node_params = hipMemcpy3DParms{};
                std::memset(&copy_proj_to_expanded_node_params, 0, sizeof(hipMemcpy3DParms));
                copy_proj_to_expanded_node_params.srcPos = make_hipPos(0, 0, 0);
                copy_proj_to_expanded_node_params.srcPtr = make_hipPitchedPtr(
                    static_cast<void*>(proj), proj_pitch, proj_geom.N_h, proj_geom.N_v);
                copy_proj_to_expanded_node_params.dstPos = make_hipPos(0, 0, 0);
                copy_proj_to_expanded_node_params.dstPtr = make_hipPitchedPtr(
                    static_cast<void*>(expanded), expanded_pitch, proj_geom.N_hFFT, proj_geom.N_v);
                copy_proj_to_expanded_node_params.extent = make_hipExtent(
                    proj_geom.N_h * sizeof(float), proj_geom.N_v, 1);
                copy_proj_to_expanded_node_params.kind = hipMemcpyDeviceToDevice;
                hipGraphNode_t copy_proj_to_expanded_deps[] = {expanded_memset_node, weighting_kernel_node};
                auto copy_proj_to_expanded_node = hipGraphNode_t{};
                hip_check(hipGraphAddMemcpyNode(
                    &copy_proj_to_expanded_node,
                    graph,
                    copy_proj_to_expanded_deps,
                    2,
                    &copy_proj_to_expanded_node_params
                ));

                // R2C Fourier-transform projection
                auto transformed = transformed_projections.at(branch_idx);
                auto transformed_pitch = transformed_pitches.at(branch_idx);
                auto transformed_memset_node_params = hipMemsetParams{};
                transformed_memset_node_params.dst = static_cast<void*>(transformed);
                transformed_memset_node_params.elementSize = sizeof(float); // elementSize maximum is 4 bytes
                transformed_memset_node_params.height = proj_geom.N_v;
                transformed_memset_node_params.pitch = transformed_pitch;
                transformed_memset_node_params.value = 0;
                transformed_memset_node_params.width = proj_geom.N_hTrans * 2; // hipfftComplex = 2 floats
                auto transformed_memset_node = hipGraphNode_t{};
                hip_check(hipGraphAddMemsetNode(
                    &transformed_memset_node, graph, nullptr, 0, &transformed_memset_node_params
                ));

                auto& forward = forward_plans.at(branch_idx);
                hipGraphNode_t forward_deps[] = {transformed_memset_node, copy_proj_to_expanded_node};
                hip_check(hipStreamBeginCaptureToGraph(stream,
                                                       graph,
                                                       forward_deps,
                                                       nullptr,
                                                       2,
                                                       hipStreamCaptureModeGlobal));
                hipfft_check(hipfftExecR2C(forward, expanded, transformed));
                hip_check(hipStreamEndCapture(stream, &graph));

                // Get last node in graph -> this is the hipFFT operation's node
                auto nodes = std::vector<hipGraphNode_t>{};
                auto num_nodes = std::size_t{};
                hip_check(hipGraphGetNodes(graph, nullptr, &num_nodes));
                nodes.resize(num_nodes);
                hip_check(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
                auto r2c_node = nodes.back();

                // Apply filter
                auto filter_blocks_per_grid = dim3{(proj_geom.N_hTrans / threads_per_block.x) + 1,
                                                   (proj_geom.N_v / threads_per_block.y) + 1,
                                                   1};
                void* filter_application_kernel_params[] =
                {
                    static_cast<void*>(&transformed), static_cast<void*>(&transformed_pitch),
                    static_cast<void*>(&R),
                    static_cast<void*>(&proj_geom.N_hTrans), static_cast<void*>(&proj_geom.N_v)
                };
                auto filter_application_kernel_node_params = hipKernelNodeParams{};
                filter_application_kernel_node_params.blockDim = threads_per_block;
                filter_application_kernel_node_params.extra = nullptr;
                filter_application_kernel_node_params.func = reinterpret_cast<void*>(filter_application_kernel);
                filter_application_kernel_node_params.gridDim = filter_blocks_per_grid;
                filter_application_kernel_node_params.kernelParams = filter_application_kernel_params;
                filter_application_kernel_node_params.sharedMemBytes = 0;
                auto filter_application_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &filter_application_kernel_node, graph, &r2c_node, 1, &filter_application_kernel_node_params
                ));

                // C2R Fourier-transform projection
                auto& backward = backward_plans.at(branch_idx);
                hip_check(hipStreamBeginCaptureToGraph(
                    stream, graph, &filter_application_kernel_node, nullptr, 1, hipStreamCaptureModeGlobal
                ));
                hipfft_check(hipfftExecC2R(backward, transformed, expanded));
                hip_check(hipStreamEndCapture(stream, &graph));

                hip_check(hipGraphGetNodes(graph, nullptr, &num_nodes));
                nodes.resize(num_nodes);
                hip_check(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
                auto c2r_node = nodes.back();

                // Shrink projection to original size and normalize
                auto copy_expanded_to_proj_node_params = hipMemcpy3DParms{};
                std::memset(&copy_expanded_to_proj_node_params, 0, sizeof(hipMemcpy3DParms));
                copy_expanded_to_proj_node_params.srcPos = make_hipPos(0, 0, 0);
                copy_expanded_to_proj_node_params.srcPtr = make_hipPitchedPtr(
                    static_cast<void*>(expanded), expanded_pitch, proj_geom.N_hFFT, proj_geom.N_v
                );
                copy_expanded_to_proj_node_params.dstPos = make_hipPos(0, 0, 0);
                copy_expanded_to_proj_node_params.dstPtr = make_hipPitchedPtr(
                    static_cast<void*>(proj), proj_pitch, proj_geom.N_h, proj_geom.N_v
                );
                copy_expanded_to_proj_node_params.extent = make_hipExtent(
                    proj_geom.N_h * sizeof(float), proj_geom.N_v, 1);
                copy_expanded_to_proj_node_params.kind = hipMemcpyDeviceToDevice;
                auto copy_expanded_to_proj_node = hipGraphNode_t{};
                hip_check(hipGraphAddMemcpyNode(
                    &copy_expanded_to_proj_node, graph, &c2r_node, 1, &copy_expanded_to_proj_node_params
                ));

                void* filter_normalization_kernel_params[] =
                {
                    static_cast<void*>(&proj), static_cast<void*>(&proj_pitch),
                    static_cast<void*>(&proj_geom.N_hFFT),
                    static_cast<void*>(&proj_geom.N_h), static_cast<void*>(&proj_geom.N_v)
                };
                auto filter_normalization_kernel_node_params = hipKernelNodeParams{};
                filter_normalization_kernel_node_params.blockDim = threads_per_block;
                filter_normalization_kernel_node_params.extra = nullptr;
                filter_normalization_kernel_node_params.func = reinterpret_cast<void*>(filter_normalization_kernel);
                filter_normalization_kernel_node_params.gridDim = blocks_per_grid;
                filter_normalization_kernel_node_params.kernelParams = filter_normalization_kernel_params;
                filter_normalization_kernel_node_params.sharedMemBytes = 0;
                auto filter_normalization_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &filter_normalization_kernel_node,
                    graph,
                    &copy_expanded_to_proj_node,
                    1,
                    &filter_normalization_kernel_node_params));

                // Backprojection
                auto theta_deg = proj_geom.theta_sign * proj_geom.theta_step * proj_idx; // Current angle
                auto theta_rad = theta_deg * std::numbers::pi_v<float> / 180.f; // Convert to radians
                auto sin_theta = std::sin(theta_rad);
                auto cos_theta = std::cos(theta_rad);

                auto bp_block_size = dim3{32u, 8u, 4u};
                auto bp_blocks = dim3{static_cast<std::uint32_t>(vol_geom.N_x / bp_block_size.x + 1),
                                      static_cast<std::uint32_t>(vol_geom.N_y / bp_block_size.y + 1),
                                      static_cast<std::uint32_t>(s_N_z / bp_block_size.z + 1)};

                // The backprojection always runs in the first branch. If multiple branches are preprocessing, the
                // additional branches need to wait for preceding backprojection kernels.
                if(branch_idx == 0)
                    hip_check(hipGraphAddEmptyNode(&preceding_bp_kernel_node, graph, nullptr, 0));

                hipGraphNode_t backprojection_kernel_deps[] =
                {
                    filter_normalization_kernel_node,
                    preceding_bp_kernel_node
                };

                auto& p_tex = texture_projections.at(branch_idx);

                auto z_offset = s_N_z_first * sub_idx;
                void* backprojection_kernel_params[] =
                {
                    &vol_ptr.ptr,
                    static_cast<void*>(&vol_ptr.pitch),
                    static_cast<void*>(&vol_geom.N_x),
                    static_cast<void*>(&vol_geom.N_y),
                    static_cast<void*>(&vol_geom.N_z),
                    static_cast<void*>(&s_N_z),
                    static_cast<void*>(&z_offset),
                    static_cast<void*>(&vol_geom.d_x),
                    static_cast<void*>(&vol_geom.d_y),
                    static_cast<void*>(&vol_geom.d_z),
                    static_cast<void*>(&p_tex),
                    static_cast<void*>(&h_min),
                    static_cast<void*>(&v_min),
                    static_cast<void*>(&sin_theta),
                    static_cast<void*>(&cos_theta),
                    static_cast<void*>(&d_h),
                    static_cast<void*>(&d_v),
                    static_cast<void*>(&d_sd),
                    static_cast<void*>(&d_so)
                };
                auto backprojection_kernel_node_params = hipKernelNodeParams{};
                backprojection_kernel_node_params.blockDim = bp_block_size;
                backprojection_kernel_node_params.extra = nullptr;
                backprojection_kernel_node_params.func = reinterpret_cast<void*>(backprojection_kernel);
                backprojection_kernel_node_params.gridDim = bp_blocks;
                backprojection_kernel_node_params.kernelParams = backprojection_kernel_params;
                backprojection_kernel_node_params.sharedMemBytes = 0;
                auto backprojection_kernel_node = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &backprojection_kernel_node,
                    graph,
                    backprojection_kernel_deps,
                    2,
                    &backprojection_kernel_node_params
                ));

                // The backprojection always runs in the first branch. If multiple branches are preprocessing, the
                // additional branches need to wait for preceding backprojection kernels.
                if(branch_idx > 0)
                    preceding_bp_kernel_node = backprojection_kernel_node;

                if(branch_idx == (num_branches - 1))
                {
                    // We have created the last branch and can now create the graph.
                    if(!graphCreated)
                    {
                        hip_check(hipGraphDebugDotPrint(graph, "graph.dot", hipGraphDebugDotFlagsVerbose));

                        // If this is the first iteration, we need to instantiate a new executable graph
                        hip_check(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
                        hip_check(hipGraphDestroy(graph));
                        graphCreated = true;
                    }

                    else
                    {
                        // Update existing executable graph after each iteration with new input data
                        auto result = hipGraphExecUpdateResult{};
                        auto errorNode = hipGraphNode_t{};
                        hip_check(hipGraphExecUpdate(graphExec, graph, &errorNode, &result));
                        if(result != hipGraphExecUpdateSuccess)
                        {
                            auto msg = std::string{"Failed to update graph: "};
                            switch(result)
                            {
                            case hipGraphExecUpdateError:
                                msg += "Invalid value.";
                                break;
                            case hipGraphExecUpdateErrorFunctionChanged:
                                msg += "Function of kernel node changed.";
                                break;
                            case hipGraphExecUpdateErrorNodeTypeChanged:
                                msg += "Type of node changed.";
                                break;
                            case hipGraphExecUpdateErrorNotSupported:
                                msg += "Something about the node is not supported.";
                                break;
                            case hipGraphExecUpdateErrorParametersChanged:
                                msg += "Unsupported parameter change.";
                                break;
                            case hipGraphExecUpdateErrorTopologyChanged:
                                msg += "Graph topology changed.";
                                break;
                            case hipGraphExecUpdateErrorUnsupportedFunctionChange:
                                msg += "Unsupported change of kernel node function.";
                                break;
                            default:
                                msg += "Unknown error.";
                                break;
                            }
                            throw std::runtime_error{msg};
                        }
                        hip_check(hipGraphDestroy(graph));
                    }
                }
                
                // Launch the graph
                hip_check(hipGraphLaunch(graphExec, graphStream));
                hip_check(hipEventRecord(graphEvent, graphStream));

                branch_idx = (branch_idx + 1) % num_branches;
                ++proj_idx;
            }

            auto memcpy_params = hipMemcpy3DParms{};
            std::memset(&memcpy_params, 0, sizeof(hipMemcpy3DParms));
            memcpy_params.dstPos = make_hipPos(0, 0, 0);
            memcpy_params.dstPtr = host_vol_ptr;
            memcpy_params.srcPos = make_hipPos(0, 0, 0);
            memcpy_params.srcPtr = vol_ptr;
            memcpy_params.extent = vol_ext;
            memcpy_params.kind = hipMemcpyDeviceToHost;
            hip_check(hipStreamWaitEvent(streams.at(0), graphEvent));
            hip_check(hipMemcpy3DAsync(&memcpy_params, streams.at(0)));
            hip_check(hipEventRecord(save_event, streams.at(0)));
            
            auto save_vol_args = new save_volume_args
            {
                "volume.tif",
                host_vol_raw_ptr,
                vol_geom.N_x, vol_geom.N_y, vol_geom.N_z, s_N_z,
                vol_geom.d_x, vol_geom.d_y
            };
            hip_check(hipStreamWaitEvent(save_stream, save_event));
            hip_check(hipLaunchHostFunc(save_stream, save_volume, save_vol_args));
        }

        std::cout << "All work items enqueued, waiting for completion...";
        hip_check(hipStreamSynchronize(save_stream));
        std::cout << "Done!" << std::endl;

        hip_check(hipFree(vol_ptr.ptr));
        hip_check(hipFreeHost(host_vol_raw_ptr));

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