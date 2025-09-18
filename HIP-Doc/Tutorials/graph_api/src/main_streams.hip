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
#include "dataset.hpp"
#include "filtering.hpp"
#include "log_transform.hpp"
#include "normalization.hpp"
#include "phantom.hpp"
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

auto main(int argc, char** argv) -> int
{
    try
    {
        auto useDataset = false;
        if(argc > 1)
        {
            if(argc > 2)
            {
                std::cerr << "Too many options passed. --dataset is the only valid option." << std::endl;
                return EXIT_FAILURE;
            }

            if(std::strcmp(argv[1], "--dataset") != 0)
            {
                std::cerr << "Invalid option passed. --dataset is the only valid option." << std::endl;
                return EXIT_FAILURE;
            }
            
            useDataset = true;
        }

        // Fetch device properties
        auto devProps = hipDeviceProp_t{};
        hip_check(hipGetDeviceProperties(&devProps, 0));

        auto const numStreams = devProps.asyncEngineCount;

        std::cout << "Device has " << numStreams << " asynchronous engines; preprocessing will use "
                  << numStreams << " parallel streams." << std::endl;

        auto streams = std::vector<hipStream_t>{};
        streams.resize(numStreams);
        for(auto&& stream : streams)
            hip_check(hipStreamCreate(&stream));

        auto projGeom = projectionGeometry{};
        auto volGeom = volumeGeometry{};
        auto numSubvols = int{}; // number of subvolumes
        auto sub_N_z  = std::vector<std::size_t>{}; // number of voxels in z direction for each subvolume

        auto r = static_cast<float*>(nullptr);
        auto R = static_cast<hipfftComplex*>(nullptr);

        auto forwardPlans = std::vector<hipfftHandle>{};
        auto forwardSizes = std::vector<std::size_t>{};
        auto backwardPlans = std::vector<hipfftHandle>{};
        auto backwardSizes = std::vector<std::size_t>{};
        forwardPlans.resize(numStreams);
        forwardSizes.resize(numStreams);
        backwardPlans.resize(numStreams);
        backwardSizes.resize(numStreams);

        auto hostInputs = std::vector<std::uint16_t*>{};
        auto inputProjections = std::vector<std::uint16_t*>{};
        auto inputPitches = std::vector<std::size_t>{};
        auto projections = std::vector<float*>{};
        auto projectionPitches = std::vector<std::size_t>{};
        auto expandedProjections = std::vector<float*>{};
        auto expandedPitches = std::vector<std::size_t>{};
        auto transformedProjections = std::vector<hipfftComplex*>{};
        auto transformedPitches = std::vector<std::size_t>{};
        auto textureProjections = std::vector<hipTextureObject_t>{};

        auto preprocessingEvents = std::vector<hipEvent_t>{};
        auto backprojectionEvents = std::vector<hipEvent_t>{};
        
        auto saveStream = hipStream_t{};
        auto saveEvent = hipEvent_t{};

        // A directory iterator is not guaranteed to go through the elements alphabetically (or another way of sorting).
        // Using a set for the paths guarantees alphabetical ordering.
        auto paths = std::set<std::filesystem::path>{};
        if(useDataset)
        {
            auto const dataPath = std::filesystem::path{DATA_DIRECTORY};
            auto dataIt = std::filesystem::directory_iterator{dataPath};
            for(auto& entry : dataIt)
                paths.insert(entry.path());
        }

        auto phantomProjections = std::vector<hipPitchedPtr>{};
        if(!useDataset)
        {
            projGeom = phantom::make_projectionGeometry();
            volGeom = phantom::make_volumeGeometry();
            phantomProjections = phantom::make_projections(projGeom, volGeom);
        }
        else
        {
            // Load first projection and obtain geometry from there
            for(auto const& path : paths)
            {
                if(path.extension() != std::filesystem::path{".tif"})
                    continue;
                else
                {
                    projGeom = dataset::make_projectionGeometry(path.string());
                    volGeom = dataset::make_volumeGeometry(projGeom);
                    break;
                }
            }
        }

        std::cout << "Initializing... ";

        auto stream = streams.at(0);

        // Create filter kernel
        hip_check(hipMalloc(reinterpret_cast<void**>(&r), projGeom.dimFFT.x * sizeof(float)));
        auto const creationBlocks = std::max((projGeom.dimFFT.x / 1024u), 1u);
        filter_creation_kernel<<<creationBlocks, 1024, 0, stream>>>(r, projGeom.s_dimFFT.x, projGeom.pixelDim.x);

        hip_check(hipMalloc(reinterpret_cast<void**>(&R), projGeom.dimTrans.x * sizeof(hipfftComplex)));
        auto filterPlan = hipfftHandle{};
        hipfft_check(hipfftPlan1d(&filterPlan, projGeom.dimFFT.x, HIPFFT_R2C, 1));
        hipfft_check(hipfftSetStream(filterPlan, stream));
        hipfft_check(hipfftExecR2C(filterPlan, r, R));

        auto absoluteBlocks = (projGeom.dimTrans.x / 1024u) + 1u;
        filter_absolute_kernel<<<absoluteBlocks, 1024, 0, stream>>>(R, projGeom.dimTrans.x, projGeom.pixelDim.x);

        hip_check(hipStreamSynchronize(stream));
                
        hipfft_check(hipfftDestroy(filterPlan));
        hip_check(hipFree(r));

        /* Estimate number of subvolumes. We will split the volume along the z axis */
        auto const inputProjSingle = projGeom.dim.x * projGeom.dim.y * sizeof(std::uint16_t);
        auto const inputProjTotal = inputProjSingle * 2u * numStreams;
        auto const projSingle = projGeom.dim.x * projGeom.dim.y * sizeof(float);
        auto const projTotal = projSingle * numStreams;
        auto const expandedSingle = projGeom.dimFFT.x * projGeom.dimFFT.y * sizeof(float);
        auto const expandedTotal = expandedSingle * numStreams;
        auto const transformedSingle = projGeom.dimTrans.x * projGeom.dimTrans.y * sizeof(hipfftComplex);
        auto const transformedTotal = transformedSingle * numStreams;
        auto const volumeTotal = volGeom.dim.x * volGeom.dim.y * volGeom.dim.z * sizeof(float);
        auto const memTotal = inputProjTotal + projTotal + expandedTotal + transformedTotal + volumeTotal;
        auto devMemFree = std::size_t{};
        auto devMemTotal = std::size_t{};
        hip_check(hipMemGetInfo(&devMemFree, &devMemTotal));

        auto memRequired = static_cast<std::size_t>(memTotal);
        numSubvols = 1;
        while(std::max(memRequired, devMemFree) == memRequired)
        {
            memRequired /= 2;
            numSubvols *= 2;
        }

        // if there is less than a GiB of free memory left on the device, use smaller subvolumes to be safe
        if((devMemFree - memRequired) < (1024 * 1024 * 1024))
            numSubvols *= 2;

        auto const s_N_z = volGeom.dim.z / numSubvols;
        for(auto i = 0; i < numSubvols - 1; ++i)
            sub_N_z.push_back(s_N_z);

        if(auto const remainder = volGeom.dim.z % numSubvols; remainder == 0)
            sub_N_z.push_back(s_N_z);
        else
            sub_N_z.push_back(s_N_z + remainder);

        hip_check(hipStreamCreate(&saveStream));
        hip_check(hipEventCreate(&saveEvent));

        std::cout << "Done!" << std::endl;
        std::cout << "Volume dimensions: " << volGeom.dim.x << " x "
                                           << volGeom.dim.y << " x "
                                           << volGeom.dim.z << std::endl;
        if(numSubvols > 1)
        {
            std::cout << "Number of subvolumes: " << numSubvols << std::endl;
            for(auto s_N_z : sub_N_z)
                std::cout << "\tDimensions: " << volGeom.dim.x << " x "
                                              << volGeom.dim.y << " x "
                                              << s_N_z << std::endl;
        }

        // Initialize per-stream data
        for(auto streamIdx = 0; streamIdx < streams.size(); ++streamIdx)
        {
            std::cout << "Initializing stream " << streamIdx << "... ";
            auto stream = streams.at(streamIdx);

            if(useDataset)
            {
                auto hostInput = static_cast<std::uint16_t*>(nullptr);
                hip_check(hipHostMalloc(reinterpret_cast<void**>(&hostInput),
                                        projGeom.dim.x * projGeom.dim.y * sizeof(std::uint16_t),
                                        hipHostMallocWriteCombined));
                hostInputs.push_back(hostInput);

                auto input = static_cast<std::uint16_t*>(nullptr);
                auto inputPitch = std::size_t{};
                hip_check(hipMallocPitch(
                    reinterpret_cast<void**>(&input),
                    &inputPitch,
                    projGeom.dim.x * sizeof(std::uint16_t),
                    projGeom.dim.y
                ));
                inputProjections.push_back(input);
                inputPitches.push_back(inputPitch);
            }

            auto proj = static_cast<float*>(nullptr);
            auto projPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&proj), &projPitch, projGeom.dim.x * sizeof(float), projGeom.dim.y
            ));
            projections.push_back(proj);
            projectionPitches.push_back(projPitch);

            auto expanded = static_cast<float*>(nullptr);
            auto expandedPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&expanded),
                &expandedPitch,
                projGeom.dimFFT.x * sizeof(float),
                projGeom.dimFFT.y
            ));
            expandedProjections.push_back(expanded);
            expandedPitches.push_back(expandedPitch);
            
            auto transformed = static_cast<hipfftComplex*>(nullptr);
            auto transformedPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&transformed),
                &transformedPitch,
                projGeom.dimTrans.x * sizeof(hipfftComplex),
                projGeom.dimTrans.y
            ));
            transformedProjections.push_back(transformed);
            transformedPitches.push_back(transformedPitch);

            auto& forward = forwardPlans.at(streamIdx);
            auto& forwardSize = forwardSizes.at(streamIdx);
            auto fw_inembed = static_cast<int>(expandedPitch / sizeof(float));
            auto fw_istride = 1;
            auto fw_idist = fw_inembed;
            auto fw_onembed = static_cast<int>(transformedPitch / sizeof(hipfftComplex));
            auto fw_ostride = 1;
            auto fw_odist = fw_onembed;
            hipfft_check(hipfftCreate(&forward));
            hipfft_check(hipfftMakePlanMany(forward, 1, &projGeom.s_dimFFT.x,
                                            &fw_inembed, 1, fw_idist,
                                            &fw_onembed, 1, fw_odist,
                                            HIPFFT_R2C, projGeom.s_dimFFT.y, &forwardSize));
            hipfft_check(hipfftSetStream(forward, stream));

            auto& backward = backwardPlans.at(streamIdx);
            auto& backwardSize = backwardSizes.at(streamIdx);
            auto bw_inembed = fw_onembed;
            auto bw_istride = fw_ostride;
            auto bw_idist = fw_odist;
            auto bw_onembed = fw_inembed;
            auto bw_ostride = fw_istride;
            auto bw_odist = fw_idist;
            hipfft_check(hipfftCreate(&backward));
            hipfft_check(hipfftMakePlanMany(backward, 1, &projGeom.s_dimFFT.x,
                                            &bw_inembed, bw_istride, bw_idist,
                                            &bw_onembed, bw_ostride, bw_odist,
                                            HIPFFT_C2R, projGeom.s_dimFFT.y, &backwardSize));
            hipfft_check(hipfftSetStream(backward, stream));

            // create a HIP texture from the projection
            auto resDesc = hipResourceDesc{};
            resDesc.resType = hipResourceTypePitch2D;
            resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();
            resDesc.res.pitch2D.devPtr = static_cast<void*>(proj);
            resDesc.res.pitch2D.width = projGeom.dim.x;
            resDesc.res.pitch2D.height = projGeom.dim.y;
            resDesc.res.pitch2D.pitchInBytes = projPitch;

            auto texDesc = hipTextureDesc{};
            texDesc.addressMode[0] = hipAddressModeBorder;
            texDesc.addressMode[1] = hipAddressModeBorder;
            texDesc.readMode = hipReadModeElementType;
            texDesc.borderColor[0] = 0.f;
            texDesc.borderColor[0] = 0.f;
            texDesc.filterMode = hipFilterModeLinear;
            texDesc.normalizedCoords = 0;

            auto& projTex = textureProjections.emplace_back();
            hip_check(hipCreateTextureObject(&projTex, &resDesc, &texDesc, nullptr));

            if(streamIdx > 0)
            {
                auto ppEvent = hipEvent_t{};
                hip_check(hipEventCreate(&ppEvent));
                preprocessingEvents.push_back(ppEvent);

                auto bpEvent = hipEvent_t{};
                hip_check(hipEventCreate(&bpEvent));
                backprojectionEvents.push_back(bpEvent);
            }

            std::cout << "Done!" << std::endl;
        }

        auto const s_N_z_first = sub_N_z.front(); // Use the initial subvolume size for offset calculations
        auto const s_N_z_last = sub_N_z.back(); // If there is a largest subvolume, it will be the last
        create_volume("volume.tif");
        auto hostVolPtr = static_cast<float*>(nullptr);
        hip_check(hipHostMalloc(
            reinterpret_cast<void**>(&hostVolPtr),
            volGeom.dim.x * volGeom.dim.y * s_N_z_last * sizeof(float),
            hipHostMallocDefault
        ));
        auto hostVol = make_hipPitchedPtr(
            hostVolPtr, volGeom.dim.x * sizeof(float), volGeom.dim.x, volGeom.dim.y
        );    
        auto vol = hipPitchedPtr{};
        auto volExt = make_hipExtent(volGeom.dim.x * sizeof(float), volGeom.dim.y, s_N_z_last);
        hip_check(hipMalloc3D(&vol, volExt));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MAIN LOOP
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(auto subIdx = 0; subIdx < numSubvols; ++subIdx)
        {
            auto s_N_z = sub_N_z.at(subIdx);

            auto subIdxInfo = new int;
            *subIdxInfo = subIdx;
            hip_check(hipLaunchHostFunc(streams.at(0), [](void* arg)
            {
                auto mySubIdx = static_cast<int*>(arg);
                std::cout << "Processing subvolume #" << *mySubIdx << std::endl;
                delete mySubIdx;
            }, subIdxInfo));
            hip_check(hipMemset3DAsync(vol, 0, volExt, streams.at(0)));

            auto pathsIt = std::begin(paths);
            auto projIdx = 0;
            while(projIdx < projGeom.numProj)
            {
                auto batchSize = std::min(numStreams, static_cast<int>(projGeom.numProj - projIdx));

                // Launch batch in parallel streams
                for(auto streamIdx = 0; streamIdx < batchSize; ++streamIdx, ++projIdx)
                {
                    auto stream = streams.at(streamIdx);

                    auto input = static_cast<std::uint16_t*>(nullptr);
                    auto inputPitch = std::size_t{};
                    if(!useDataset)
                    {
                        auto inputPitchedPtr = phantomProjections.at(projIdx);
                        input = static_cast<std::uint16_t*>(inputPitchedPtr.ptr);
                        inputPitch = inputPitchedPtr.pitch;
                    }
                    else
                    {
                        // Skip non-tif files
                        while(pathsIt != std::end(paths) && pathsIt->extension() != std::filesystem::path{".tif"})
                            ++pathsIt;

                        if(pathsIt == std::end(paths))
                        {
                            std::cerr << "Warning: Unexpected end of input files." << std::endl;
                            break;
                        }

                        // Using a pointer here ensures that the arguments live beyond the scope of a loop iteration.
                        // The pointer will be deleted inside load_projection().
                        auto path = *pathsIt;
                        auto loadProjArgs = new load_projection_args
                        {
                            path.string(),
                            projGeom.dim.x,
                            projGeom.dim.y,
                            hostInputs.at(streamIdx)
                        };
                        hip_check(hipLaunchHostFunc(stream, load_projection, static_cast<void*>(loadProjArgs)));

                        auto hostInput = hostInputs.at(streamIdx);
                        input = inputProjections.at(streamIdx);
                        inputPitch = inputPitches.at(streamIdx);
                        hip_check(hipMemcpy2DAsync(
                            input,
                            inputPitch,
                            hostInput,
                            projGeom.dim.x * sizeof(std::uint16_t),
                            projGeom.dim.x * sizeof(std::uint16_t),
                            projGeom.dim.y,
                            hipMemcpyHostToDevice,
                            stream
                        ));
                    }

                    auto threadsPerBlock = dim3{32, 32, 1};
                    auto blocksPerGrid = dim3{
                        (projGeom.dim.x / threadsPerBlock.x) + 1, (projGeom.dim.y / threadsPerBlock.y) + 1, 1
                    };

                    auto proj = projections.at(streamIdx);
                    auto projPitch = projectionPitches.at(streamIdx);
                    normalization_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        input, inputPitch, proj, projPitch, projGeom.dim, projGeom.bps
                    );

                    log_transformation_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        proj, projPitch, projGeom.dim
                    );

                    weighting_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        proj,
                        projPitch,
                        projGeom.dim,
                        projGeom.d_sd,
                        projGeom.d_so,
                        projGeom.minCoord,
                        projGeom.pixelDim
                    );

                    // Expand projection to filter length
                    auto expanded = expandedProjections.at(streamIdx);
                    auto expandedPitch = expandedPitches.at(streamIdx);
                    hip_check(hipMemset2DAsync(
                        expanded, expandedPitch, 0, projGeom.dimFFT.x * sizeof(float), projGeom.dimFFT.y, stream
                    ));
                    hip_check(hipMemcpy2DAsync(
                        expanded,
                        expandedPitch,
                        proj,
                        projPitch,
                        projGeom.dim.x * sizeof(float),
                        projGeom.dim.y,
                        hipMemcpyDeviceToDevice,
                        stream
                    ));

                    // R2C Fourier-transform projection
                    auto transformed = transformedProjections.at(streamIdx);
                    auto transformedPitch = transformedPitches.at(streamIdx);
                    hip_check(hipMemset2DAsync(
                        transformed,
                        transformedPitch,
                        0,
                        projGeom.dimTrans.x * sizeof(hipfftComplex),
                        projGeom.dimTrans.y,
                        stream
                    ));
                    auto& forward = forwardPlans.at(streamIdx);
                    hipfft_check(hipfftExecR2C(forward, expanded, transformed));
                    
                    // Apply filter
                    auto filterBlocksPerGrid = dim3{
                        (projGeom.dimTrans.x / threadsPerBlock.x) + 1,
                        (projGeom.dimTrans.y / threadsPerBlock.y) + 1,
                        1
                    };
                    filter_application_kernel<<<filterBlocksPerGrid, threadsPerBlock, 0, stream>>>(
                        transformed, transformedPitch, R, projGeom.dimTrans
                    );
                    
                    auto& backward = backwardPlans.at(streamIdx);
                    hipfft_check(hipfftExecC2R(backward, transformed, expanded));

                    // Shrink projection to original size and normalize
                    hip_check(hipMemcpy2DAsync(
                        proj,
                        projPitch,
                        expanded,
                        expandedPitch,
                        projGeom.dim.x * sizeof(float),
                        projGeom.dim.y,
                        hipMemcpyDeviceToDevice,
                        stream
                    ));
                    
                    filter_normalization_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        proj, projPitch, projGeom.dimFFT.x, projGeom.dim
                    );

                    // The backprojection always runs in the first stream. If multiple streams are preprocessing, let
                    // the first stream wait for the current projection before starting the backprojection.
                    if(streamIdx > 0)
                    {
                        auto ppEvent = preprocessingEvents.at(streamIdx - 1);
                        hip_check(hipEventRecord(ppEvent, stream));
                        hip_check(hipStreamWaitEvent(streams.at(0), ppEvent));
                    }

                    // Backprojection
                    auto thetaDeg = projGeom.thetaSign * projGeom.thetaStep * projIdx; // Current angle
                    auto thetaRad = thetaDeg * std::numbers::pi_v<float> / 180.f; // Convert to radians
                    auto sinTheta = std::sin(thetaRad);
                    auto cosTheta = std::cos(thetaRad);

                    auto bpBlockSize = dim3{32u, 8u, 4u};
                    auto bpBlocks = dim3{
                        static_cast<std::uint32_t>(volGeom.dim.x / bpBlockSize.x + 1),
                        static_cast<std::uint32_t>(volGeom.dim.y / bpBlockSize.y + 1),
                        static_cast<std::uint32_t>(s_N_z / bpBlockSize.z + 1)
                    };

                    auto& projTex = textureProjections.at(streamIdx);
                    backprojection_kernel<<<bpBlocks, bpBlockSize, 0, streams.at(0)>>>(
                        static_cast<float*>(vol.ptr),
                        vol.pitch,
                        volGeom.dim,
                        s_N_z,
                        s_N_z_first * subIdx,
                        volGeom.voxelDim,
                        projTex,
                        projGeom.minCoord,
                        sinTheta,
                        cosTheta,
                        projGeom.pixelDim,
                        projGeom.d_sd,
                        projGeom.d_so
                    );

                    // The backprojection always runs in the first stream. If multiple streams are preprocessing, let
                    // the current stream wait for the backprojection to finish before reusing the projection memory.
                    if(streamIdx > 0)
                    {
                        auto bpEvent = backprojectionEvents.at(streamIdx - 1);
                        hip_check(hipEventRecord(bpEvent, streams.at(0)));
                        hip_check(hipStreamWaitEvent(stream, bpEvent));
                    }

                    if(useDataset)
                        ++pathsIt;
                }
            }
            
            // Copy volume back to host and save
            auto memcpyParams = hipMemcpy3DParms{};
            std::memset(&memcpyParams, 0, sizeof(hipMemcpy3DParms));
            memcpyParams.dstPos = make_hipPos(0, 0, 0);
            memcpyParams.dstPtr = hostVol;
            memcpyParams.srcPos = make_hipPos(0, 0, 0);
            memcpyParams.srcPtr = vol;
            memcpyParams.extent = volExt;
            memcpyParams.kind = hipMemcpyDeviceToHost;
            hip_check(hipMemcpy3DAsync(&memcpyParams, streams.at(0)));
            hip_check(hipEventRecord(saveEvent, streams.at(0)));
            
            auto saveVolArgs = new save_volume_args
            {
                "volume.tif",
                hostVolPtr,
                volGeom.dim.x, volGeom.dim.y, volGeom.dim.z, s_N_z,
                volGeom.voxelDim.x, volGeom.voxelDim.y
            };
            hip_check(hipStreamWaitEvent(saveStream, saveEvent));
            hip_check(hipLaunchHostFunc(saveStream, save_volume, saveVolArgs));
        }

        std::cout << "All work items enqueued, waiting for completion...";
        hip_check(hipStreamSynchronize(saveStream));
        std::cout << "Done!" << std::endl;

        hip_check(hipFree(vol.ptr));
        hip_check(hipFreeHost(hostVolPtr));

        for(auto&& e : backprojectionEvents)
            hip_check(hipEventDestroy(e));

        for(auto&& e : preprocessingEvents)
            hip_check(hipEventDestroy(e));

        for(auto&& tex : textureProjections)
            hip_check(hipDestroyTextureObject(tex));

        for(auto&& plan : backwardPlans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& plan : forwardPlans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& p : transformedProjections)
            hip_check(hipFree(p));

        for(auto&& p : expandedProjections)
            hip_check(hipFree(p));

        for(auto&& p : projections)
            hip_check(hipFree(p));

        if(useDataset)
        {
            for(auto&& p : inputProjections)
                hip_check(hipFree(p));

            for(auto&& p : hostInputs)
                hip_check(hipFreeHost(p));
        }
        else
        {
            for(auto&& p : phantomProjections)
                hip_check(hipFree(p.ptr));
        }

        hip_check(hipEventDestroy(saveEvent));
        hip_check(hipStreamDestroy(saveStream));

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
