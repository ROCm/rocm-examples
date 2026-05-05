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

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesMatmul.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "CmdParser/cmdparser.hpp"
#include "hipdnn_utils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename ComputeType>
bool runMatmul(hipdnnHandle_t handle, bool cpuValidation)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running matmul graph " << inputType
              << (cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t batch = 2;
    constexpr int64_t m     = 3;
    constexpr int64_t k     = 4;
    constexpr int64_t n     = 5;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(inputType)
        .set_compute_data_type(DataType::FLOAT);

    utilities::Tensor<InputType> aTensor({batch, m, k});
    utilities::Tensor<InputType> bTensor({batch, k, n});

    auto aAttr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("A", inputType, aTensor));
    auto bAttr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("B", inputType, bTensor));

    graph::MatmulAttributes matmulAttrs;
    matmulAttrs.set_name("matmul_node");

    auto cAttr = graph->matmul(aAttr, bAttr, matmulAttrs);
    cAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> cTensor(cAttr->get_dim());

    aTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    bTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    cTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[aAttr->get_uid()] = aTensor.memory().deviceData();
    variantPack[bAttr->get_uid()] = bTensor.memory().deviceData();
    variantPack[cAttr->get_uid()] = cTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    cTensor.memory().markDeviceModified();

    auto cHostPtr = cTensor.memory().hostData();

    std::cout << "First 10 C values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(cHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> cRefTensor(cAttr->get_dim());

        hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<InputType,
                                                                  InputType,
                                                                  InputType,
                                                                  ComputeType>(
            aTensor, bTensor, cRefTensor);

        auto tolerance
            = hipdnn_test_sdk::utilities::matmul::calculateMatmulTolerance<InputType, InputType, ComputeType>(
                aTensor, bTensor);

        auto cValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        bool cValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "C", cValidator, cRefTensor, cTensor, tolerance, tolerance);

        validationPassed = cValid;
    }

    std::cout << "Matmul graph execution complete for " << inputType << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<bool>("vc", "verify-cpu", false, "Enable CPU reference validation");
    parser.run_and_exit_if_error();
    const bool cpuValidation = parser.get<bool>("vc");

    auto [handle, handleError] = createHipdnnHandle();
    HIPDNN_FE_CHECK(handleError);

    bool allPassed = true;
    allPassed &= runMatmul<float, float>(*handle, cpuValidation);
    allPassed &= runMatmul<half, float>(*handle, cpuValidation);
    allPassed &= runMatmul<bfloat16, float>(*handle, cpuValidation);

    if(allPassed)
    {
        std::cout << "All matmul runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more matmul runs failed validation.\n";
        return 1;
    }
}
