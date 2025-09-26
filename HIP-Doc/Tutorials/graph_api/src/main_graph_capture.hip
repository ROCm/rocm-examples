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
#include "phantom.hpp"
#include "projection.hpp"
#include "utility.hpp"
#include "weighting.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numbers>
#include <set>
#include <stdexcept>
#include <vector>

auto main() -> int
{
    try
    {
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

        auto inputProjections = std::vector<std::uint16_t*>{};
        auto inputPitches = std::vector<std::size_t>{};
        auto projections = std::vector<float*>{};
        auto projectionPitches = std::vector<std::size_t>{};
        auto expandedProjections = std::vector<float*>{};
        auto expandedPitches = std::vector<std::size_t>{};
        auto transformedProjections = std::vector<hipfftComplex*>{};
        auto transformedPitches = std::vector<std::size_t>{};
        auto textureProjections = std::vector<hipTextureObject_t>{};
        
        auto saveStream = hipStream_t{};
        auto saveEvent = hipEvent_t{};

        auto projGeom = phantom::make_projectionGeometry();
        auto volGeom = phantom::make_volumeGeometry();
        auto phantomProjections = phantom::make_projections(projGeom, volGeom);
        
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
        auto const inputProjTotal = inputProjSingle * numStreams;
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
        auto graphCreated = false;
        auto graph = hipGraph_t{};
        auto graphExec = hipGraphExec_t{};
        auto graphStream = hipStream_t{};

        hip_check(hipStreamCreate(&graphStream));
        auto start = std::chrono::steady_clock::now();
        for(auto subIdx = 0; subIdx < numSubvols; ++subIdx)
        {
            auto s_N_z = sub_N_z.at(subIdx);

            auto subIdxInfo = new int;
            *subIdxInfo = subIdx;
            hip_check(hipLaunchHostFunc(graphStream, [](void* arg)
            {
                auto mySubIdx = static_cast<int*>(arg);
                std::cout << "Processing subvolume #" << *mySubIdx << std::endl;
                delete mySubIdx;
            }, subIdxInfo));
            hip_check(hipMemset3DAsync(vol, 0, volExt, graphStream));

            auto projIdx = 0;
            while(projIdx < projGeom.numProj)
            {
                // Capture the current batch
                hip_check(hipStreamBeginCapture(streams.at(0), hipStreamCaptureModeGlobal));

                auto batchSize = std::min(numStreams, static_cast<int>(projGeom.numProj - projIdx));

                // Fork: Record events on stream 0, then have other streams wait
                for(auto streamIdx = 1; streamIdx < batchSize; ++streamIdx)
                {
                    auto forkEvent = hipEvent_t{};
                    hip_check(hipEventCreate(&forkEvent));
                    hip_check(hipEventRecord(forkEvent, streams.at(0)));
                    hip_check(hipStreamWaitEvent(streams.at(streamIdx), forkEvent));
                    hip_check(hipEventDestroy(forkEvent)); // Can destroy after wait is recorded
                }

                // Launch batch in parallel streams
                for(auto streamIdx = 0; streamIdx < batchSize; ++streamIdx, ++projIdx)
                {
                    auto stream = streams.at(streamIdx);

                    auto threadsPerBlock = dim3{32, 32, 1};
                    auto blocksPerGrid = dim3{
                        (projGeom.dim.x / threadsPerBlock.x) + 1, (projGeom.dim.y / threadsPerBlock.y) + 1, 1
                    };

                    auto inputPitchedPtr = phantomProjections.at(projIdx);
                    auto input = static_cast<std::uint16_t*>(inputPitchedPtr.ptr);
                    auto inputPitch = inputPitchedPtr.pitch;

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
                    backprojection_kernel<<<bpBlocks, bpBlockSize, 0, stream>>>(
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
                }

                // Join: Record events on all streams except stream 0, then have stream 0 wait
                for(auto streamIdx = 1; streamIdx < batchSize; ++streamIdx)
                {
                    auto joinEvent = hipEvent_t{};
                    hip_check(hipEventCreate(&joinEvent));
                    hip_check(hipEventRecord(joinEvent, streams.at(streamIdx)));
                    hip_check(hipStreamWaitEvent(streams.at(0), joinEvent));
                    hip_check(hipEventDestroy(joinEvent)); // Can destroy after wait is recorded
                }

                // Stop capturing -- this will stop capturing on all streams
                hip_check(hipStreamEndCapture(streams.at(0), &graph));

                // Instantiate and launch the graph
                if(batchSize == numStreams)
                {
                    if(!graphCreated)
                    {
                        hip_check(hipGraphDebugDotPrint(graph, "graph_capture.dot", hipGraphDebugDotFlagsVerbose));

                        hip_check(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
                        hip_check(hipGraphDestroy(graph));
                        hip_check(hipGraphLaunch(graphExec, graphStream));
                        graphCreated = true;
                    }
                    else
                    {
                        // Make sure preceding graph has finished
                        hip_check(hipStreamSynchronize(graphStream));

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
                        hip_check(hipGraphLaunch(graphExec, graphStream));
                    }
                }
                else
                {
                    hip_check(hipGraphDebugDotPrint(graph, "graph_capture_final.dot", hipGraphDebugDotFlagsVerbose));

                    // Make sure preceding graphes have finished
                    hip_check(hipStreamSynchronize(graphStream));
                    hip_check(hipGraphExecDestroy(graphExec));

                    // Incomplete batch: topology changed, must instantiate new executable graph
                    hip_check(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
                    hip_check(hipGraphDestroy(graph));
                    hip_check(hipGraphLaunch(graphExec, graphStream));
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
            hip_check(hipMemcpy3DAsync(&memcpyParams, graphStream));
            hip_check(hipEventRecord(saveEvent, graphStream));
            
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

        std::cout << "All work items enqueued, waiting for completion... ";
        hip_check(hipStreamSynchronize(saveStream));
        auto stop = std::chrono::steady_clock::now();
        std::cout << "Done!" << std::endl;
        auto const elapsed = std::chrono::duration<double>{stop - start};
        std::cout << "Reconstruction time: " << elapsed << std::endl;

        // Cleanup
        hip_check(hipGraphExecDestroy(graphExec));
        hip_check(hipStreamDestroy(graphStream));

        hip_check(hipFree(vol.ptr));
        hip_check(hipFreeHost(hostVolPtr));

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

        for(auto&& p : phantomProjections)
            hip_check(hipFree(p.ptr));

        hip_check(hipEventDestroy(saveEvent));
        hip_check(hipStreamDestroy(saveStream));

        hip_check(hipFree(R));

        for(auto&& stream : streams)
            hip_check(hipStreamDestroy(stream));

        return EXIT_SUCCESS;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
