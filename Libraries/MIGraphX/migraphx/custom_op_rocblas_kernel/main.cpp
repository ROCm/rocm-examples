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
#include <rocblas/rocblas.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

/// \brief Checks if the provided rocBLAS status is success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define ROCBLAS_CHECK(condition)                                                         \
    {                                                                                    \
        const rocblas_status status = condition;                                         \
        if(status != rocblas_status_success)                                             \
        {                                                                                \
            std::cerr << "rocBLAS error: " << rocblas_status_to_string(status) << " at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                       \
            std::exit(error_exit_code);                                                  \
        }                                                                                \
    }

rocblas_handle create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    return rocblas_handle{handle};
}

rocblas_handle create_rocblas_handle_ptr(migraphx::context& ctx)
{
    HIP_CHECK(hipSetDevice(0));
    rocblas_handle rb     = create_rocblas_handle_ptr();
    auto*          stream = ctx.get_queue<hipStream_t>();
    ROCBLAS_CHECK(rocblas_set_stream(rb, stream));
    return rb;
}

struct sscal_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override
    {
        return "sscal_custom_op";
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
        (void)output_shape;
        // Create rocBLAS stream handle
        auto rb_handle = create_rocblas_handle_ptr(ctx);
        ROCBLAS_CHECK(rocblas_set_pointer_mode(rb_handle, rocblas_pointer_mode_device));
        rocblas_int n       = args[1].get_shape().lengths()[0];
        float*      alpha   = reinterpret_cast<float*>(args[0].data());
        float*      vec_ptr = reinterpret_cast<float*>(args[1].data());
        ROCBLAS_CHECK(rocblas_sscal(rb_handle, n, alpha, vec_ptr, 1));
        ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
        return args[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("sscal_custom_op must have 2 input arguments");
        }
        if(inputs[0].lengths().size() != 1 || inputs[0].lengths()[0] != 1)
        {
            throw std::runtime_error("first input argument to sscal_custom_op must be a scalar");
        }
        if(inputs[1].lengths().size() != 1)
        {
            throw std::runtime_error(
                "second input argument to sscal_custom_op must be a vector with dimension one");
        }
        return inputs.back();
    }
};

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("device", "device", 0, "Device ID to use");
    parser.set_optional<int>("size", "size", 8192, "Vector size");
    parser.set_optional<float>("scale", "scale", -1.0f, "Scale factor");
    parser.run_and_exit_if_error();

    // Get arguments
    const int   device_id = parser.get<int>("device");
    const int   size      = parser.get<int>("size");
    const float scale     = parser.get<float>("scale");

    if(size <= 0)
    {
        std::cerr << "Error: Size must be positive" << std::endl;
        return error_exit_code;
    }

    // Set device
    HIP_CHECK(hipSetDevice(device_id));

    // Computes ReLU(neg(x) * scale)
    sscal_custom_op sscal_op;
    migraphx::register_experimental_custom_op(sscal_op);

    // Build program
    migraphx::program p;
    migraphx::shape   x_shape{migraphx_shape_float_type, {static_cast<size_t>(size)}};
    migraphx::shape   scale_shape{migraphx_shape_float_type, {1}};
    migraphx::module  mod         = p.get_main_module();
    auto              x           = mod.add_parameter("x", x_shape);
    auto              scale_param = mod.add_parameter("scale", scale_shape);
    auto              neg_ins     = mod.add_instruction(migraphx::operation("neg"), {x});
    auto              custom_kernel
        = mod.add_instruction(migraphx::operation("sscal_custom_op"), {scale_param, neg_ins});
    auto relu_ins = mod.add_instruction(migraphx::operation("relu"), {custom_kernel});
    mod.add_return({relu_ins});

    // Compile program
    migraphx::compile_options options;
    // Set offload copy to true for GPUs
    options.set_offload_copy();
    MIGRAPHX_CHECK(p.compile(migraphx::target("gpu"), options));

    // Prepare input data
    migraphx::program_parameters pp;
    std::vector<float>           x_data(x_shape.elements());
    std::vector<float>           scale_data{scale};
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(x_shape, x_data.data()));
    pp.add("scale", migraphx::argument(scale_shape, scale_data.data()));

    // Execute program
    auto results = p.eval(pp);
    auto result  = results[0];

    // Verify result
    std::vector<float> expected_result = x_data;
    if(bool{result == migraphx::argument(x_shape, expected_result.data())})
    {
        std::cout << "Successfully executed custom rocBLAS kernel example" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Custom rocBLAS kernel example failed" << std::endl;
        return error_exit_code;
    }
}
