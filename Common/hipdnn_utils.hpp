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

#ifndef COMMON_HIPDNN_UTILS_HPP
#define COMMON_HIPDNN_UTILS_HPP

#include "example_utils.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define HIPDNN_CHECK(status)                                                             \
    {                                                                                    \
        const hipdnnStatus_t _status = (status);                                         \
        if(_status != HIPDNN_STATUS_SUCCESS)                                             \
        {                                                                                \
            std::cerr << "hipDNN Error: " << hipdnnGetErrorString(_status) << " at "     \
                      << __FILE__ << ':' << __LINE__ << std::endl;                       \
            std::exit(error_exit_code);                                                  \
        }                                                                                \
    }

#define HIPDNN_FE_CHECK(statusObj)                                                       \
    {                                                                                    \
        auto const& _status = (statusObj);                                               \
        if(!_status.is_good())                                                           \
        {                                                                                \
            std::cerr << "hipDNN Frontend Error: " << _status.get_message() << " at "    \
                      << __FILE__ << ':' << __LINE__ << std::endl;                       \
            std::exit(error_exit_code);                                                  \
        }                                                                                \
    }

// Skip-aware variant of HIPDNN_FE_CHECK for use inside bool-returning sample
// callbacks. When graph->build() fails the macro prints a skip message and
// returns true so the enclosing variant is counted as gracefully skipped
// rather than hard-failing the entire sample. This handles cases where no
// engine has an applicable solution for a given dtype/layout on the current
// device. This macro contains `return true;` and MUST only be used in a
// bool-returning function context.
#define HIPDNN_FE_CHECK_SKIPPABLE(statusObj)                                                      \
    {                                                                                             \
        auto const& _status = (statusObj);                                                        \
        if(!_status.is_good())                                                                    \
        {                                                                                         \
            if(_status.get_code() == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED)             \
            {                                                                                     \
                std::cout << "Skipping: no engine has an applicable solution for this "           \
                          << "graph on the current device. (" << _status.get_message() << ")\n";  \
                return true;                                                                      \
            }                                                                                     \
            std::cerr << "hipDNN Frontend Error: " << _status.get_message() << " at "             \
                      << __FILE__ << ':' << __LINE__ << std::endl;                                \
            std::exit(error_exit_code);                                                           \
        }                                                                                         \
    }

using hipdnn_data_sdk::utilities::TensorLayout;

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

template <typename F>
bool run(F&& f)
{
    bool allPassed = true;
    allPassed &= f.template operator()<float, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<half, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<float, float>(TensorLayout::NHWC);
    allPassed &= f.template operator()<half, float>(TensorLayout::NHWC);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::NHWC);
    return allPassed;
}

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    createTensor(const std::vector<int64_t>& dims,
                 hipdnn_frontend::DataType_t dataType,
                 const TensorLayout& layout = TensorLayout::NCHW)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(dims).set_data_type(dataType);
    tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder));

    return tensor;
}

inline int64_t
    getTensorElementCount(const std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& tensor)
{
    int64_t count = 1;
    for(auto dim : tensor->get_dim())
    {
        count *= dim;
    }
    return count;
}

struct SampleRunner
{
    hipdnnHandle_t handle;
    bool cpuValidation;
    bool useRunningStats = false;

    template <typename InputType, typename IntermediateType>
    bool operator()(const TensorLayout& layout);
};

#endif // COMMON_HIPDNN_UTILS_HPP
