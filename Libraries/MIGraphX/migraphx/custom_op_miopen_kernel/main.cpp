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

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "migraphx_utils.hpp"

#include <hip/hip_runtime.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

/// \brief Checks if the provided MIOpen status is success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define MIOPEN_CHECK(condition)                                                          \
    {                                                                                    \
        const miopenStatus_t status = condition;                                         \
        if(status != miopenStatusSuccess)                                                \
        {                                                                                \
            std::cerr << "MIOpen error at " << __FILE__ << ':' << __LINE__ << std::endl; \
            std::exit(error_exit_code);                                                  \
        }                                                                                \
    }

inline miopenTensorDescriptor_t make_miopen_tensor(const migraphx::shape& s)
{
    miopenTensorDescriptor_t t;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&t));
    // Convert to ints
    auto             s_lens = s.lengths();
    std::vector<int> lens(s_lens.begin(), s_lens.end());
    auto             s_strides = s.strides();
    std::vector<int> strides(s_strides.begin(), s_strides.end());
    miopenDataType_t d;
    if(s.type() == migraphx_shape_float_type)
    {
        d = miopenFloat;
    }
    else if(s.type() == migraphx_shape_half_type)
    {
        d = miopenHalf;
    }
    else if(s.type() == migraphx_shape_int32_type)
    {
        d = miopenInt32;
    }
    else if(s.type() == migraphx_shape_int8_type)
    {
        d = miopenInt8;
    }
    else
    {
        throw std::runtime_error("MAKE_TENSOR: unsupported type");
    }
    miopenSetTensorDescriptor(t, d, s_lens.size(), lens.data(), strides.data());
    return t;
}

inline auto make_miopen_handle(migraphx::context& ctx)
{
    HIP_CHECK(hipSetDevice(0));
    auto*          stream = ctx.get_queue<hipStream_t>();
    miopenHandle_t out;
    MIOPEN_CHECK(miopenCreateWithStream(&out, stream));
    return out;
}

inline auto make_activation_descriptor(miopenActivationMode_t mode,
                                       double                 alpha = 0,
                                       double                 beta  = 0,
                                       double                 gamma = 0)
{
    miopenActivationDescriptor_t ad;
    MIOPEN_CHECK(miopenCreateActivationDescriptor(&ad));
    miopenSetActivationDescriptor(ad, mode, alpha, beta, gamma);
    return ad;
}

struct abs_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override
    {
        return "abs_custom_op";
    }

    // Flag to identify whether custom op runs on the GPU or on the host.
    // Based on this flag MIGraphX would inject necessary copies to and from GPU for the input and
    // output buffers as necessary. Therefore if custom_op runs on GPU then it can assume its input
    // buffers are in GPU memory, and similarly for the host
    virtual bool runs_on_offload_target() const override
    {
        return true;
    }

    virtual migraphx::argument compute(migraphx::context   ctx,
                                       migraphx::shape     output_shape,
                                       migraphx::arguments args) const override
    {
        float alpha = 1;
        float beta  = 0;
        // MIOpen kernel call takes raw buffer pointers for the TensorData. These Buffer pointers
        // must be accompanied with Tensor Description e.g. shape, type, strides, dimensionality.
        // Following `make_miopen_tensor` makes such tensor descriptors to pass as parameter to
        // MIOpen kernel call.
        auto y_desc = make_miopen_tensor(output_shape);
        auto x_desc = make_miopen_tensor(args[0].get_shape());
        // Create MIOpen stream handle
        auto miopen_handle = make_miopen_handle(ctx);
        // MIOpen has generic kernel for many different kinds of activation functions.
        // Each such generic call must be accompanied with description of what kind of activation
        // computation to perform
        auto ad = make_activation_descriptor(miopenActivationABS, 0, 0, 0);
        miopenActivationForward(miopen_handle,
                                ad,
                                &alpha,
                                x_desc,
                                args[0].data(),
                                &beta,
                                y_desc,
                                args[1].data());
        return args[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("abs_custom_op must have two input arguments");
        }
        if(inputs[0] != inputs[1])
        {
            throw std::runtime_error("Input arguments to abs_custom_op must have same shape");
        }
        return inputs.back();
    }
};

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("device", "device", 0, "Device ID to use");
    parser.set_optional<int>("m", "m", 32, "First dimension size");
    parser.set_optional<int>("n", "n", 256, "Second dimension size");
    parser.run_and_exit_if_error();

    // Get arguments
    const int device_id = parser.get<int>("device");
    const int m         = parser.get<int>("m");
    const int n         = parser.get<int>("n");

    if(m <= 0 || n <= 0)
    {
        std::cerr << "Error: Dimensions must be positive" << std::endl;
        return error_exit_code;
    }

    // Set device
    HIP_CHECK(hipSetDevice(device_id));

    // Register custom operation
    abs_custom_op abs_op;
    migraphx::register_experimental_custom_op(abs_op);

    // Build program
    migraphx::program p;
    migraphx::shape   s{
        migraphx_shape_float_type,
          {static_cast<size_t>(m), static_cast<size_t>(n)}
    };
    migraphx::module mod     = p.get_main_module();
    auto             x       = mod.add_parameter("x", s);
    auto             neg_ins = mod.add_instruction(migraphx::operation("neg"), {x});
    // Add allocation for the custom_kernel's output buffer
    auto alloc = mod.add_allocation(s);
    auto custom_kernel
        = mod.add_instruction(migraphx::operation("abs_custom_op"), {neg_ins, alloc});
    auto relu_ins = mod.add_instruction(migraphx::operation("relu"), {custom_kernel});
    mod.add_return({relu_ins});

    // Compile program
    migraphx::compile_options options;
    // Set offload copy to true for GPUs
    options.set_offload_copy();
    MIGRAPHX_CHECK(p.compile(migraphx::target("gpu"), options));

    // Prepare input data
    migraphx::program_parameters prog_params;
    std::vector<float>           x_data(s.bytes() / sizeof(s.type()));
    std::iota(x_data.begin(), x_data.end(), 0);
    prog_params.add("x", migraphx::argument(s, x_data.data()));

    // Execute program
    auto results = p.eval(prog_params);
    auto result  = results[0];

    // Verify result
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::abs(i); });

    if(bool{result == migraphx::argument(s, expected_result.data())})
    {
        std::cout << "Successfully executed custom MIOpen kernel example with MIGraphX"
                  << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Custom MIOpen kernel example failed" << std::endl;
        return error_exit_code;
    }
}
