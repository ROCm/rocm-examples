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

        auto const numBranches = devProps.asyncEngineCount;

        std::cout << "Device supports " << numBranches << " asynchronous engines; preprocessing will use "
                  << numBranches << " parallel branches." << std::endl;
        
        // For interoperability with hipFFT we require streams
        auto streams = std::vector<hipStream_t>{};
        streams.resize(numBranches);
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
        forwardPlans.resize(numBranches);
        forwardSizes.resize(numBranches);
        backwardPlans.resize(numBranches);
        backwardSizes.resize(numBranches);

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
        auto const inputProjTotal = inputProjSingle * 2u * numBranches;
        auto const projSingle = projGeom.dim.x * projGeom.dim.y * sizeof(float);
        auto const projTotal = projSingle * numBranches;
        auto const expandedSingle = projGeom.dimFFT.x * projGeom.dimFFT.y * sizeof(float);
        auto const expandedTotal = expandedSingle * numBranches;
        auto const transformedSingle = projGeom.dimTrans.x * projGeom.dimTrans.y * sizeof(hipfftComplex);
        auto const transformedTotal = transformedSingle * numBranches;
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

        // Initialize per-branch data
        for(auto branchIdx = 0; branchIdx < numBranches; ++branchIdx)
        {
            std::cout << "Initializing branch #" << branchIdx << "...";
            auto stream = streams.at(branchIdx);

            if(useDataset)
            {
                auto hostInput = static_cast<std::uint16_t*>(nullptr);
                hip_check(hipHostMalloc(
                    reinterpret_cast<void**>(&hostInput),
                    projGeom.dim.x * projGeom.dim.y * sizeof(std::uint16_t),
                    hipHostMallocWriteCombined
                ));
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

            auto& forward = forwardPlans.at(branchIdx);
            auto& forwardSize = forwardSizes.at(branchIdx);
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

            auto& backward = backwardPlans.at(branchIdx);
            auto& backwardSize = backwardSizes.at(branchIdx);
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

            auto pathsIt = std::begin(paths);
            auto projIdx = 0;
            while(projIdx < projGeom.numProj)
            {
                // Create graph for this batch
                auto graph = hipGraph_t{};
                hip_check(hipGraphCreate(&graph, 0));

                auto batchSize = std::min(numBranches, static_cast<int>(projGeom.numProj - projIdx));

                // Add nodes for each projection in batch
                for(auto branchIdx = 0; branchIdx < batchSize; ++branchIdx, ++projIdx)
                {
                    auto stream = streams.at(branchIdx);

                    auto input = static_cast<std::uint16_t*>(nullptr);
                    auto inputPitch = std::size_t{};
                    auto copyInputNode = hipGraphNode_t{};
                    if(!useDataset)
                    {
                        auto inputPitchedPtr = phantomProjections.at(projIdx);
                        input = static_cast<std::uint16_t*>(inputPitchedPtr.ptr);
                        inputPitch = inputPitchedPtr.pitch;
                        hip_check(hipGraphAddEmptyNode(&copyInputNode, graph, nullptr, 0));
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
                            hostInputs.at(branchIdx)
                        };
                        auto loadProjNodeParams = hipHostNodeParams{};
                        loadProjNodeParams.fn = load_projection;
                        loadProjNodeParams.userData = static_cast<void*>(loadProjArgs);
                        auto loadProjNode = hipGraphNode_t{};
                        hip_check(hipGraphAddHostNode(&loadProjNode, graph, nullptr, 0, &loadProjNodeParams));

                        auto hostInput = hostInputs.at(branchIdx);
                        input = inputProjections.at(branchIdx);
                        inputPitch = inputPitches.at(branchIdx);
                        auto copyInputNodeParams = hipMemcpy3DParms{};
                        std::memset(&copyInputNodeParams, 0, sizeof(hipMemcpy3DParms));
                        copyInputNodeParams.srcPos = make_hipPos(0, 0, 0);
                        copyInputNodeParams.srcPtr = make_hipPitchedPtr(
                            static_cast<void*>(hostInput),
                            projGeom.dim.x * sizeof(std::uint16_t),
                            projGeom.dim.x,
                            projGeom.dim.y
                        );
                        copyInputNodeParams.dstPos = make_hipPos(0, 0, 0);
                        copyInputNodeParams.dstPtr = make_hipPitchedPtr(
                            static_cast<void*>(input), inputPitch, projGeom.dim.x, projGeom.dim.y
                        );
                        copyInputNodeParams.extent = make_hipExtent(
                            projGeom.dim.x * sizeof(std::uint16_t), projGeom.dim.y, 1
                        );
                        copyInputNodeParams.kind = hipMemcpyHostToDevice;
                        hip_check(hipGraphAddMemcpyNode(
                            &copyInputNode, graph, &loadProjNode, 1, &copyInputNodeParams
                        ));
                    }

                    auto threadsPerBlock = dim3{32, 32, 1};
                    auto blocksPerGrid = dim3{
                        (projGeom.dim.x / threadsPerBlock.x) + 1, (projGeom.dim.y / threadsPerBlock.y) + 1, 1
                    };

                    auto proj = projections.at(branchIdx);
                    auto projPitch = projectionPitches.at(branchIdx);
                    void* normalizationKernelParams[] =
                    {
                        static_cast<void*>(&input),
                        static_cast<void*>(&inputPitch),
                        static_cast<void*>(&proj),
                        static_cast<void*>(&projPitch),
                        static_cast<void*>(&projGeom.dim),
                        static_cast<void*>(&projGeom.bps)
                    };

                    auto normalizationKernelNodeParams = hipKernelNodeParams{};
                    normalizationKernelNodeParams.blockDim = threadsPerBlock;
                    normalizationKernelNodeParams.extra = nullptr;
                    normalizationKernelNodeParams.func = reinterpret_cast<void*>(normalization_kernel);
                    normalizationKernelNodeParams.gridDim = blocksPerGrid;
                    normalizationKernelNodeParams.kernelParams = normalizationKernelParams;
                    normalizationKernelNodeParams.sharedMemBytes = 0;
                    auto normalizationKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &normalizationKernelNode, graph, &copyInputNode, 1, &normalizationKernelNodeParams
                    ));

                    void* logTransformationKernelParams[] =
                    {
                        static_cast<void*>(&proj),
                        static_cast<void*>(&projPitch),
                        static_cast<void*>(&projGeom.dim)
                    };
                    auto logTransformationKernelNodeParams = hipKernelNodeParams{};
                    logTransformationKernelNodeParams.blockDim = threadsPerBlock;
                    logTransformationKernelNodeParams.extra = nullptr;
                    logTransformationKernelNodeParams.func = reinterpret_cast<void*>(log_transformation_kernel);
                    logTransformationKernelNodeParams.gridDim = blocksPerGrid;
                    logTransformationKernelNodeParams.kernelParams = logTransformationKernelParams;
                    logTransformationKernelNodeParams.sharedMemBytes = 0;
                    auto logTransformationKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &logTransformationKernelNode,
                        graph,
                        &normalizationKernelNode,
                        1,
                        &logTransformationKernelNodeParams
                    ));

                    void* weightingKernelParams[] =
                    {
                        static_cast<void*>(&proj),
                        static_cast<void*>(&projPitch),
                        static_cast<void*>(&projGeom.dim),
                        static_cast<void*>(&projGeom.d_sd),
                        static_cast<void*>(&projGeom.d_so),
                        static_cast<void*>(&projGeom.minCoord),
                        static_cast<void*>(&projGeom.pixelDim)
                    };
                    auto weightingKernelNodeParams = hipKernelNodeParams{};
                    weightingKernelNodeParams.blockDim = threadsPerBlock;
                    weightingKernelNodeParams.extra = nullptr;
                    weightingKernelNodeParams.func = reinterpret_cast<void*>(weighting_kernel);
                    weightingKernelNodeParams.gridDim = blocksPerGrid;
                    weightingKernelNodeParams.kernelParams = weightingKernelParams;
                    weightingKernelNodeParams.sharedMemBytes = 0;
                    auto weightingKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &weightingKernelNode, graph, &logTransformationKernelNode, 1, &weightingKernelNodeParams
                    ));

                    // Expand projection to filter length
                    auto expanded = expandedProjections.at(branchIdx);
                    auto expandedPitch = expandedPitches.at(branchIdx);
                    auto expandedMemsetNodeParams = hipMemsetParams{};
                    expandedMemsetNodeParams.dst = static_cast<void*>(expanded);
                    expandedMemsetNodeParams.elementSize = sizeof(float);
                    expandedMemsetNodeParams.height = projGeom.dimFFT.y;
                    expandedMemsetNodeParams.pitch = expandedPitch;
                    expandedMemsetNodeParams.value = 0;
                    expandedMemsetNodeParams.width = projGeom.dimFFT.x;
                    auto expandedMemsetNode = hipGraphNode_t{};
                    hip_check(hipGraphAddMemsetNode(
                        &expandedMemsetNode, graph, nullptr, 0, &expandedMemsetNodeParams
                    ));

                    auto copyProjToexpandedNodeParams = hipMemcpy3DParms{};
                    std::memset(&copyProjToexpandedNodeParams, 0, sizeof(hipMemcpy3DParms));
                    copyProjToexpandedNodeParams.srcPos = make_hipPos(0, 0, 0);
                    copyProjToexpandedNodeParams.srcPtr = make_hipPitchedPtr(
                        static_cast<void*>(proj), projPitch, projGeom.dim.x, projGeom.dim.y);
                    copyProjToexpandedNodeParams.dstPos = make_hipPos(0, 0, 0);
                    copyProjToexpandedNodeParams.dstPtr = make_hipPitchedPtr(
                        static_cast<void*>(expanded), expandedPitch, projGeom.dimFFT.x, projGeom.dimFFT.y);
                    copyProjToexpandedNodeParams.extent = make_hipExtent(
                        projGeom.dim.x * sizeof(float), projGeom.dim.y, 1);
                    copyProjToexpandedNodeParams.kind = hipMemcpyDeviceToDevice;
                    hipGraphNode_t copyProjToexpandedDeps[] = {expandedMemsetNode, weightingKernelNode};
                    auto copyProjToexpandedNode = hipGraphNode_t{};
                    hip_check(hipGraphAddMemcpyNode(
                        &copyProjToexpandedNode,
                        graph,
                        copyProjToexpandedDeps,
                        2,
                        &copyProjToexpandedNodeParams
                    ));

                    // R2C Fourier-transform projection
                    auto transformed = transformedProjections.at(branchIdx);
                    auto transformedPitch = transformedPitches.at(branchIdx);
                    auto transformedMemsetNodeParams = hipMemsetParams{};
                    transformedMemsetNodeParams.dst = static_cast<void*>(transformed);
                    transformedMemsetNodeParams.elementSize = sizeof(float); // elementSize maximum is 4 bytes
                    transformedMemsetNodeParams.height = projGeom.dimTrans.y;
                    transformedMemsetNodeParams.pitch = transformedPitch;
                    transformedMemsetNodeParams.value = 0;
                    transformedMemsetNodeParams.width = projGeom.dimTrans.x * 2; // hipfftComplex = 2 floats
                    auto transformedMemsetNode = hipGraphNode_t{};
                    hip_check(hipGraphAddMemsetNode(
                        &transformedMemsetNode, graph, nullptr, 0, &transformedMemsetNodeParams
                    ));

                    // Create a child graph for the forward FFT
                    auto forwardGraph = hipGraph_t{};
                    hip_check(hipGraphCreate(&forwardGraph, 0));

                    auto& forward = forwardPlans.at(branchIdx);
                    hip_check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
                    hipfft_check(hipfftExecR2C(forward, expanded, transformed));
                    hip_check(hipStreamEndCapture(stream, &forwardGraph));

                    hipGraphNode_t forwardDeps[] = {transformedMemsetNode, copyProjToexpandedNode};
                    auto forwardNode = hipGraphNode_t{};
                    hip_check(hipGraphAddChildGraphNode(&forwardNode, graph, forwardDeps, 2, forwardGraph));

                    // Apply filter
                    auto filterBlocksPerGrid = dim3{
                        (projGeom.dimTrans.x / threadsPerBlock.x) + 1,
                        (projGeom.dimTrans.y / threadsPerBlock.y) + 1,
                        1
                    };
                    void* filterApplicationKernelParams[] =
                    {
                        static_cast<void*>(&transformed),
                        static_cast<void*>(&transformedPitch),
                        static_cast<void*>(&R),
                        static_cast<void*>(&projGeom.dimTrans)
                    };
                    auto filterApplicationKernelNodeParams = hipKernelNodeParams{};
                    filterApplicationKernelNodeParams.blockDim = threadsPerBlock;
                    filterApplicationKernelNodeParams.extra = nullptr;
                    filterApplicationKernelNodeParams.func = reinterpret_cast<void*>(filter_application_kernel);
                    filterApplicationKernelNodeParams.gridDim = filterBlocksPerGrid;
                    filterApplicationKernelNodeParams.kernelParams = filterApplicationKernelParams;
                    filterApplicationKernelNodeParams.sharedMemBytes = 0;
                    auto filterApplicationKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &filterApplicationKernelNode, graph, &forwardNode, 1, &filterApplicationKernelNodeParams
                    ));

                    hip_check(hipGraphDestroy(forwardGraph));

                    // Create a child graph for the backward FFT
                    auto backwardGraph = hipGraph_t{};
                    hip_check(hipGraphCreate(&backwardGraph, 0));

                    // C2R Fourier-transform projection
                    auto& backward = backwardPlans.at(branchIdx);
                    hip_check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
                    hipfft_check(hipfftExecC2R(backward, transformed, expanded));
                    hip_check(hipStreamEndCapture(stream, &backwardGraph));

                    auto backwardNode = hipGraphNode_t{};
                    hip_check(hipGraphAddChildGraphNode(
                        &backwardNode, graph, &filterApplicationKernelNode, 1, backwardGraph
                    ));

                    // Shrink projection to original size and normalize
                    auto copyExpandedToProjNodeParams = hipMemcpy3DParms{};
                    std::memset(&copyExpandedToProjNodeParams, 0, sizeof(hipMemcpy3DParms));
                    copyExpandedToProjNodeParams.srcPos = make_hipPos(0, 0, 0);
                    copyExpandedToProjNodeParams.srcPtr = make_hipPitchedPtr(
                        static_cast<void*>(expanded), expandedPitch, projGeom.dimFFT.x, projGeom.dimFFT.y
                    );
                    copyExpandedToProjNodeParams.dstPos = make_hipPos(0, 0, 0);
                    copyExpandedToProjNodeParams.dstPtr = make_hipPitchedPtr(
                        static_cast<void*>(proj), projPitch, projGeom.dim.x, projGeom.dim.y
                    );
                    copyExpandedToProjNodeParams.extent = make_hipExtent(
                        projGeom.dim.x * sizeof(float), projGeom.dim.y, 1);
                    copyExpandedToProjNodeParams.kind = hipMemcpyDeviceToDevice;
                    auto copyExpandedToProjNode = hipGraphNode_t{};
                    hip_check(hipGraphAddMemcpyNode(
                        &copyExpandedToProjNode, graph, &backwardNode, 1, &copyExpandedToProjNodeParams
                    ));

                    hip_check(hipGraphDestroy(backwardGraph));

                    void* filterNormalizationKernelParams[] =
                    {
                        static_cast<void*>(&proj),
                        static_cast<void*>(&projPitch),
                        static_cast<void*>(&projGeom.dimFFT.x),
                        static_cast<void*>(&projGeom.dim)
                    };
                    auto filterNormalizationKernelNodeParams = hipKernelNodeParams{};
                    filterNormalizationKernelNodeParams.blockDim = threadsPerBlock;
                    filterNormalizationKernelNodeParams.extra = nullptr;
                    filterNormalizationKernelNodeParams.func = reinterpret_cast<void*>(filter_normalization_kernel);
                    filterNormalizationKernelNodeParams.gridDim = blocksPerGrid;
                    filterNormalizationKernelNodeParams.kernelParams = filterNormalizationKernelParams;
                    filterNormalizationKernelNodeParams.sharedMemBytes = 0;
                    auto filterNormalizationKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &filterNormalizationKernelNode,
                        graph,
                        &copyExpandedToProjNode,
                        1,
                        &filterNormalizationKernelNodeParams));

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

                    auto& projTex = textureProjections.at(branchIdx);

                    auto z_offset = s_N_z_first * subIdx;
                    void* backprojectionKernelParams[] =
                    {
                        &vol.ptr,
                        static_cast<void*>(&vol.pitch),
                        static_cast<void*>(&volGeom.dim),
                        static_cast<void*>(&s_N_z),
                        static_cast<void*>(&z_offset),
                        static_cast<void*>(&volGeom.voxelDim),
                        static_cast<void*>(&projTex),
                        static_cast<void*>(&projGeom.minCoord),
                        static_cast<void*>(&sinTheta),
                        static_cast<void*>(&cosTheta),
                        static_cast<void*>(&projGeom.pixelDim),
                        static_cast<void*>(&projGeom.d_sd),
                        static_cast<void*>(&projGeom.d_so)
                    };
                    auto backprojectionKernelNodeParams = hipKernelNodeParams{};
                    backprojectionKernelNodeParams.blockDim = bpBlockSize;
                    backprojectionKernelNodeParams.extra = nullptr;
                    backprojectionKernelNodeParams.func = reinterpret_cast<void*>(backprojection_kernel);
                    backprojectionKernelNodeParams.gridDim = bpBlocks;
                    backprojectionKernelNodeParams.kernelParams = backprojectionKernelParams;
                    backprojectionKernelNodeParams.sharedMemBytes = 0;
                    auto backprojectionKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &backprojectionKernelNode,
                        graph,
                        &filterNormalizationKernelNode,
                        1,
                        &backprojectionKernelNodeParams
                    ));

                    if(useDataset)
                        ++pathsIt;
                }

                // Instantiate and launch the graph
                if(batchSize == numBranches)
                {
                    if(!graphCreated)
                    {
                        hip_check(hipGraphDebugDotPrint(graph, "graph_creation.dot", hipGraphDebugDotFlagsVerbose));

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
                    hip_check(hipGraphDebugDotPrint(graph, "graph_creation_final.dot", hipGraphDebugDotFlagsVerbose));

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

        std::cout << "All work items enqueued, waiting for completion...";
        hip_check(hipStreamSynchronize(saveStream));
        std::cout << "Done!" << std::endl;

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
