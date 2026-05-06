/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef COMMON_HIPBLASLT_UTILS_HPP
#define COMMON_HIPBLASLT_UTILS_HPP

#include "example_utils.hpp"

#include <hipblaslt/hipblaslt.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#ifdef _WIN32
    #include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/// \brief Converts a \p hipblasStatus_t variable to its correspondent string.
inline const char* hipblas_status_to_string(hipblasStatus_t status)
{
    switch(status)
    {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
        case HIPBLAS_STATUS_INVALID_ENUM: return "HIPBLAS_STATUS_INVALID_ENUM";
        case HIPBLAS_STATUS_UNKNOWN: return "HIPBLAS_STATUS_UNKNOWN";
        // We use default because we are not in control of these enumeration values.
        // Ideally this function is something hipBLAS would provide
        default: return "<unknown hipblasStatus_t value>";
    }
}

/// \brief Checks if the provided status code is \p HIPBLAS_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPBLASLT_CHECK(condition)                                                             \
    {                                                                                          \
        const hipblasStatus_t status = (condition);                                            \
        if(status != HIPBLAS_STATUS_SUCCESS)                                                   \
        {                                                                                      \
            std::cerr << "hipBLASLt error encountered: \"" << hipblas_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;                 \
            std::exit(error_exit_code);                                                        \
        }                                                                                      \
    }

// ============================================================================
// Data Type Interface
// ============================================================================

union compute_type_interface
{
    float         f32;
    double        f64;
    hipblasLtHalf f16;
    int32_t       i32;
};

template<typename T>
constexpr auto hipblaslt_type_to_datatype()
{
    if(std::is_same_v<T, hipblasLtHalf>)
    {
        return HIP_R_16F;
    }
    if(std::is_same_v<T, hip_bfloat16>)
    {
        return HIP_R_16BF;
    }
    if(std::is_same_v<T, float>)
    {
        return HIP_R_32F;
    }
    if(std::is_same_v<T, double>)
    {
        return HIP_R_64F;
    }
    if(std::is_same_v<T, hipblaslt_f8_fnuz>)
    {
        return HIP_R_8F_E4M3_FNUZ;
    }
    if(std::is_same_v<T, hipblaslt_bf8_fnuz>)
    {
        return HIP_R_8F_E5M2_FNUZ;
    }
    if(std::is_same_v<T, hipblaslt_f8>)
    {
        return HIP_R_8F_E4M3;
    }
    if(std::is_same_v<T, hipblaslt_bf8>)
    {
        return HIP_R_8F_E5M2;
    }
    if(std::is_same_v<T, int32_t>)
    {
        return HIP_R_32I;
    }
    if(std::is_same_v<T, hipblasLtInt8>)
    {
        return HIP_R_8I;
    }

    return HIP_R_16F; // testing purposes we default to f16
}

inline hipDataType compute_type_to_real_datatype(hipblasComputeType_t ctype)
{
    static const std::map<hipblasComputeType_t, hipDataType> ctype_map{
        {          HIPBLAS_COMPUTE_16F, HIP_R_16F},
        { HIPBLAS_COMPUTE_16F_PEDANTIC, HIP_R_16F},
        {          HIPBLAS_COMPUTE_32F, HIP_R_32F},
        { HIPBLAS_COMPUTE_32F_PEDANTIC, HIP_R_32F},
        { HIPBLAS_COMPUTE_32F_FAST_16F, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_FAST_16BF, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_FAST_TF32, HIP_R_32F},
        {          HIPBLAS_COMPUTE_64F, HIP_R_64F},
        { HIPBLAS_COMPUTE_64F_PEDANTIC, HIP_R_64F},
        {          HIPBLAS_COMPUTE_32I, HIP_R_32I},
        { HIPBLAS_COMPUTE_32I_PEDANTIC, HIP_R_32I}
    };

    return ctype_map.at(ctype);
}

inline std::size_t real_datatype_size(hipDataType dtype)
{
    // These types were not defined in older versions of ROCm, so need to be handled specially here.
    auto const dtype_int = static_cast<int>(dtype);
    if(dtype_int == HIP_R_4F_E2M1 || dtype_int == HIP_R_6F_E2M3
       || dtype_int == HIP_R_6F_E3M2)
    {
        return 1;
    }

    static const std::map<hipDataType, std::size_t> dtype_map{
        {         HIP_R_32F, 4},
        {         HIP_R_64F, 8},
        {         HIP_R_16F, 2},
        {          HIP_R_8I, 1},
        {          HIP_R_8U, 1},
        {         HIP_R_32I, 4},
        {         HIP_R_32U, 4},
        {        HIP_R_16BF, 2},
        {          HIP_R_4I, 1},
        {          HIP_R_4U, 1},
        {         HIP_R_16I, 2},
        {         HIP_R_16U, 2},
        {         HIP_R_64I, 8},
        {         HIP_R_64U, 8},
        {HIP_R_8F_E4M3_FNUZ, 1},
        {HIP_R_8F_E5M2_FNUZ, 1},
        {     HIP_R_8F_E4M3, 1},
        {     HIP_R_8F_E5M2, 1},
    };

    return dtype_map.at(dtype);
}

// ============================================================================
// Tensor Data Manipulation
// ============================================================================

namespace tensor_manipulation
{
using shape_t       = std::vector<size_t>;
using strides_t     = std::vector<size_t>;
using indices_t     = std::vector<size_t>;
using permutation_t = std::vector<size_t>;

class tensor_desc
{
public:
    explicit tensor_desc(std::initializer_list<size_t> shape) : shape_(shape)
    {
        strides_.assign(shape_.size(), 1);

        for(ssize_t i = strides_.size() - 2; i >= 0; --i)
        {
            strides_[i] = strides_[i + 1] * this->shape_[i + 1];
        }
    }

    explicit tensor_desc(const shape_t& shape) : shape_(shape)
    {
        strides_.assign(shape_.size(), 1);

        for(int i = strides_.size() - 2; i >= 0; --i)
        {
            strides_[i] = strides_[i + 1] * this->shape_[i + 1];
        }
    }

    tensor_desc(std::initializer_list<size_t> shape, std::initializer_list<size_t> strides)
        : shape_(shape), strides_(strides)
    {}

    tensor_desc(const shape_t& shape, const strides_t& strides) : shape_(shape), strides_(strides)
    {}

    size_t stride(size_t i) const
    {
        return strides_.at(i);
    }

    size_t num_dims() const
    {
        return shape_.size();
    }

    size_t dim(size_t i) const
    {
        return shape_.at(i);
    }

    const shape_t& get_shape() const
    {
        return shape_;
    }

    void set_shape(const shape_t& shape)
    {
        this->shape_ = shape;
        strides_.assign(shape.size(), 1);

        for(int i = strides_.size() - 2; i >= 0; --i)
        {
            strides_[i] = strides_[i + 1] * this->shape_[i + 1];
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_desc& desc)
    {
        os << "Shape: [";
        for(auto i : desc.shape_)
        {
            os << i << ", ";
        }
        os << "]\n";
        os << "Strides: [";
        for(auto i : desc.strides_)
        {
            os << i << ", ";
        }
        os << "]\n";
        return os;
    }

    std::size_t flatten_size() const
    {
        size_t s{1};
        for(auto i : shape_)
        {
            s *= i;
        }
        return s;
    }

    bool is_shape_compatible(const shape_t& shape) const
    {
        tensor_desc new_desc(shape);
        return flatten_size() == new_desc.flatten_size();
    }

    bool can_shape_pad_to(const shape_t& shape) const
    {
        if(this->shape_.size() != shape.size())
        {
            return false;
        }

        for(size_t i = 0; i < this->shape_.size(); ++i)
        {
            if(this->shape_.at(i) > shape.at(i))
            {
                return false;
            }
        }

        return true;
    }

private:
    shape_t   shape_;
    strides_t strides_;
};

class tensor
{
public:
    template<typename T>
    static tensor create(const shape_t shape)
    {
        return tensor(shape, sizeof(T));
    }

    tensor(const shape_t shape, size_t element_size)
        : element_size_(element_size)
        , desc_(shape)
        , data_(new char[element_size * desc_.flatten_size()])
    {}

    template<typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(data_.get());
    }

    template<typename T>
    T* as()
    {
        return reinterpret_cast<T*>(data_.get());
    }

    template<typename T>
    const T& get_value(const indices_t& indices) const
    {
        size_t offset{};

        for(size_t i = 0; i < indices.size(); ++i)
        {
            const auto idx = indices[i];
            offset += desc_.stride(i) * idx;
        }

        return as<T>()[offset];
    }

    template<typename T>
    const T& set_value(const indices_t& indices, const T& value)
    {
        size_t offset{};

        for(size_t i = 0; i < indices.size(); ++i)
        {
            const auto idx = indices[i];
            offset += desc_.stride(i) * idx;
        }

        as<T>()[offset] = value;
        return value;
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor& t)
    {
        os << t.desc_;
        return os;
    }

    const tensor_desc& get_desc() const
    {
        return desc_;
    }

    size_t get_element_size() const
    {
        return element_size_;
    }

    size_t get_num_bytes() const
    {
        return get_desc().flatten_size() * get_element_size();
    }

    void reshape(const shape_t& shape)
    {
        if(desc_.is_shape_compatible(shape))
        {
            desc_.set_shape(shape);
            return;
        }
        assert(false && "Incompatible shape");
    }

private:
    size_t                  element_size_{};
    tensor_desc             desc_;
    std::unique_ptr<char[]> data_;
};

indices_t permute_indices(const indices_t& indices, const permutation_t& perm)
{
    assert(indices.size() == perm.size());
    indices_t new_indices = indices;
    for(size_t i = 0; i < perm.size(); ++i)
    {
        new_indices[i] = indices.at(perm.at(i));
    }
    return new_indices;
}

using iterate_callback_t     = std::function<void(const indices_t& indices)>;
using iterate_dim_callback_t = std::function<void(size_t dim)>;

void iterate_tensor(
    const shape_t&         shape,
    size_t                 dim,
    indices_t&             indices,
    iterate_callback_t     callback,
    iterate_dim_callback_t dim_enter_callback = [](size_t) {},
    iterate_dim_callback_t dim_leave_callback = [](size_t) {})
{
    if(dim == shape.size())
    {
        callback(indices);
        return;
    }

    dim_enter_callback(dim);

    for(size_t i = 0; i < shape.at(dim); ++i)
    {
        indices[dim] = i;
        iterate_tensor(shape, dim + 1, indices, callback, dim_enter_callback, dim_leave_callback);
    }

    dim_leave_callback(dim);
}

template<typename T>
void permute_tensor(tensor& dst, const tensor& src, const permutation_t& perm)
{
    indices_t indices(src.get_desc().num_dims(), 0);

    iterate_tensor(src.get_desc().get_shape(),
                   0,
                   indices,
                   [&dst, &src, &perm](const indices_t& indices)
                   {
                       indices_t dst_indices = permute_indices(indices, perm);
                       auto&&    value       = src.get_value<T>(indices);
                       dst.set_value<T>(dst_indices, value);
                   });
}

template<typename T>
tensor permute_tensor(const tensor& tensor_input, const permutation_t& perm)
{
    assert(tensor_input.get_desc().num_dims() == perm.size());
    assert(sizeof(T) == tensor_input.get_element_size());
    shape_t new_shape = permute_indices(tensor_input.get_desc().get_shape(), perm);
    tensor  permuted(new_shape, tensor_input.get_element_size());
    permute_tensor<T>(permuted, tensor_input, perm);
    return permuted;
}

template<typename T>
tensor pad_tensor(const tensor& src, const shape_t& new_shape, T pad_val)
{
    assert(src.get_desc().can_shape_pad_to(new_shape) && "Invalid shape for padding");
    tensor    dst(new_shape, sizeof(T));
    indices_t indices(src.get_desc().num_dims(), 0);

    iterate_tensor(dst.get_desc().get_shape(),
                   0,
                   indices,
                   [&dst, &pad_val](const indices_t& indices)
                   { dst.set_value<T>(indices, pad_val); });

    iterate_tensor(src.get_desc().get_shape(),
                   0,
                   indices,
                   [&dst, &src](const indices_t& indices)
                   {
                       auto&& value = src.get_value<T>(indices);
                       dst.set_value<T>(indices, value);
                   });
    return dst;
}

tensor pad_tensor(const tensor&  tensor_input,
                  const shape_t& new_shape,
                  const void*    pad_val_ptr,
                  size_t         pad_val_size)
{
    switch(pad_val_size)
    {
        case 1:
            return pad_tensor<uint8_t>(tensor_input,
                                       new_shape,
                                       *static_cast<const uint8_t*>(pad_val_ptr));
        case 2:
            return pad_tensor<uint16_t>(tensor_input,
                                        new_shape,
                                        *static_cast<const uint16_t*>(pad_val_ptr));
        case 4:
            return pad_tensor<uint32_t>(tensor_input,
                                        new_shape,
                                        *static_cast<const uint32_t*>(pad_val_ptr));
        case 8:
            return pad_tensor<uint64_t>(tensor_input,
                                        new_shape,
                                        *static_cast<const uint64_t*>(pad_val_ptr));
        default: assert(false && "Unsupported element size");
    }

    return tensor({0}, tensor_input.get_element_size());
}

tensor permute_tensor(const tensor& tensor_input, const permutation_t& perm)
{
    shape_t new_shape = permute_indices(tensor_input.get_desc().get_shape(), perm);
    tensor  permuted(new_shape, tensor_input.get_element_size());
    switch(tensor_input.get_element_size())
    {
        case 1: permute_tensor<uint8_t>(permuted, tensor_input, perm); break;
        case 2: permute_tensor<uint16_t>(permuted, tensor_input, perm); break;
        case 4: permute_tensor<uint32_t>(permuted, tensor_input, perm); break;
        case 8: permute_tensor<uint64_t>(permuted, tensor_input, perm); break;
        default: assert(false && "Unsupported element size");
    }
    return permuted;
}

template<typename T>
void print_tensor_data(std::ostream& os, const tensor& tensor_input)
{
    const auto* data         = tensor_input.as<T>();
    const auto  num_elements = tensor_input.get_desc().flatten_size();
    os << "[";

    for(size_t i = 0; i < num_elements; ++i)
    {
        os << float(data[i]) << ", ";
    }

    os << "]\n";
}

template<typename T>
void print_tensor_data_multi_dims(std::ostream& os, const tensor& tensor_input)
{
    os << "[";

    indices_t indices(tensor_input.get_desc().num_dims(), 0);

    iterate_tensor(
        tensor_input.get_desc().get_shape(),
        0,
        indices,
        [&os, &tensor_input](const indices_t& idx)
        { os << float(tensor_input.get_value<T>(idx)) << ", "; },
        [&os](size_t dim)
        {
            (void)dim;
            os << "[";
        },
        [&os, &tensor_input](size_t dim)
        {
            os << "], ";

            if(dim + 1 == tensor_input.get_desc().num_dims())
            {
                os << '\n';
            }
        });

    os << "]\n";
}
} // namespace tensor_manipulation

// ============================================================================
// Runner Classes
// ============================================================================

template<typename input_type_a,
         typename input_type_b,
         typename output_type,
         typename alpha_type,
         typename beta_type,
         typename bias_type = output_type>
struct runner
{
    runner(int64_t    m,
           int64_t    n,
           int64_t    k,
           int64_t    batch_count,
           alpha_type alpha,
           beta_type  beta,
           int64_t    max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        HIP_CHECK(hipStreamCreate(&stream));
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));
        HIP_CHECK(hipMalloc(&d_a, m * k * batch_count * sizeof(input_type_a)));
        HIP_CHECK(hipMalloc(&d_b, n * k * batch_count * sizeof(input_type_b)));
        HIP_CHECK(hipMalloc(&d_c, m * n * batch_count * sizeof(output_type)));
        HIP_CHECK(hipMalloc(&d_d, m * n * batch_count * sizeof(output_type)));
        HIP_CHECK(hipMalloc(&d_alpha_vec, m * batch_count * sizeof(float)));

        HIP_CHECK(hipHostMalloc(&a, m * k * batch_count * sizeof(input_type_a)));
        HIP_CHECK(hipHostMalloc(&b, n * k * batch_count * sizeof(input_type_b)));
        HIP_CHECK(hipHostMalloc(&c, m * n * batch_count * sizeof(output_type)));
        HIP_CHECK(hipHostMalloc(&d, m * n * batch_count * sizeof(output_type)));
        HIP_CHECK(hipHostMalloc(&alpha_vec, m * batch_count * sizeof(float)));

        if(max_workspace_size > 0)
            HIP_CHECK(hipMalloc(&d_workspace, max_workspace_size));

        for(int i = 0; i < m * k * batch_count; i++)
            ((input_type_a*)a)[i] = static_cast<input_type_a>((rand() % 7) - 3);
        for(int i = 0; i < n * k * batch_count; i++)
            ((input_type_b*)b)[i] = static_cast<input_type_b>((rand() % 7) - 3);
        for(int i = 0; i < m * n * batch_count; i++)
            ((output_type*)c)[i] = static_cast<output_type>((rand() % 7) - 3);
        for(int i = 0; i < m * batch_count; ++i)
            ((float*)alpha_vec)[i] = static_cast<float>((rand() % 7) - 3);
    }

    ~runner()
    {
        HIP_CHECK(hipFree(d_workspace));
        HIP_CHECK(hipFree(a));
        HIP_CHECK(hipFree(b));
        HIP_CHECK(hipFree(c));
        HIP_CHECK(hipFree(d));
        HIP_CHECK(hipFree(alpha_vec));
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        HIP_CHECK(hipFree(d_d));
        HIP_CHECK(hipFree(d_alpha_vec));
        HIPBLASLT_CHECK(hipblasLtDestroy(handle));
        HIP_CHECK(hipStreamDestroy(stream));

        if(bias_vec)
        {
            HIP_CHECK(hipFree(bias_vec));
            HIP_CHECK(hipFree(d_bias_vec));
        }
    }

    void set_bias_info(bool use_bias, char bias_src)
    {
        bias_elems = 0;
        if(use_bias)
        {
            if(bias_src == 'B' || bias_src == 'b')
                bias_elems = n;
            else if(bias_src == 'A' || bias_src == 'a' || bias_src == 'D' || bias_src == 'd')
                bias_elems = m;
            // else, bias_elems = 0
        }

        // alloc bias if use bias
        if(bias_elems > 0)
        {
            if(bias_vec)
            {
                HIP_CHECK(hipFree(bias_vec));
                HIP_CHECK(hipFree(d_bias_vec));
            }

            HIP_CHECK(hipMalloc(&d_bias_vec, bias_elems * sizeof(bias_type)));
            HIP_CHECK(hipHostMalloc(&bias_vec, bias_elems * sizeof(bias_type)));
            for(int i = 0; i < bias_elems; ++i)
                ((bias_type*)bias_vec)[i] = static_cast<bias_type>((rand() % 7) - 3);
        }
    }

    void host_to_device()
    {
        HIP_CHECK(hipMemcpyAsync(d_a,
                                 a,
                                 m * k * batch_count * sizeof(input_type_a),
                                 hipMemcpyHostToDevice,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(d_b,
                                 b,
                                 n * k * batch_count * sizeof(input_type_b),
                                 hipMemcpyHostToDevice,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(d_c,
                                 c,
                                 m * n * batch_count * sizeof(output_type),
                                 hipMemcpyHostToDevice,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(d_alpha_vec,
                                 alpha_vec,
                                 m * batch_count * sizeof(float),
                                 hipMemcpyHostToDevice,
                                 stream));

        // copy bias if needed
        if(bias_vec)
            HIP_CHECK(hipMemcpyAsync(d_bias_vec,
                                     bias_vec,
                                     bias_elems * sizeof(bias_type),
                                     hipMemcpyHostToDevice,
                                     stream));
    }

    void device_to_host()
    {
        HIP_CHECK(hipMemcpyAsync(d,
                                 d_d,
                                 m * n * batch_count * sizeof(output_type),
                                 hipMemcpyDeviceToHost,
                                 stream));
    }

    void run(const std::function<void()>& func)
    {
        host_to_device();

        static_cast<void>(func());

        device_to_host();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t    m;
    int64_t    n;
    int64_t    k;
    int64_t    batch_count;
    alpha_type alpha;
    beta_type  beta;

    void *a, *b, *c, *d, *alpha_vec; // host
    void *d_a, *d_b, *d_c, *d_d, *d_alpha_vec; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    int64_t bias_elems = 0;
    void*   bias_vec   = nullptr; // host
    void*   d_bias_vec = nullptr; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template<typename input_type_a,
         typename input_type_b,
         typename output_type,
         typename alpha_type,
         typename beta_type>
struct runner_vec
{
    runner_vec(const std::vector<int64_t>    m,
               const std::vector<int64_t>    n,
               const std::vector<int64_t>    k,
               const std::vector<int64_t>    batch_count,
               const std::vector<alpha_type> alpha,
               const std::vector<beta_type>  beta,
               const int64_t                 max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        HIP_CHECK(hipStreamCreate(&stream));
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));
        d_a.resize(m.size(), nullptr);
        d_b.resize(m.size(), nullptr);
        d_c.resize(m.size(), nullptr);
        d_d.resize(m.size(), nullptr);
        d_alpha_vec.resize(m.size(), nullptr);
        a.resize(m.size(), nullptr);
        b.resize(m.size(), nullptr);
        c.resize(m.size(), nullptr);
        d.resize(m.size(), nullptr);
        alpha_vec.resize(m.size(), nullptr);
        for(size_t j = 0; j < m.size(); j++)
        {
            HIP_CHECK(hipMalloc(&d_a[j], m[j] * k[j] * batch_count[j] * sizeof(input_type_a)));
            HIP_CHECK(hipMalloc(&d_b[j], n[j] * k[j] * batch_count[j] * sizeof(input_type_b)));
            HIP_CHECK(hipMalloc(&d_c[j], m[j] * n[j] * batch_count[j] * sizeof(output_type)));
            HIP_CHECK(hipMalloc(&d_d[j], m[j] * n[j] * batch_count[j] * sizeof(output_type)));
            HIP_CHECK(hipMalloc(&d_alpha_vec[j], m[j] * batch_count[j] * sizeof(float)));

            HIP_CHECK(hipHostMalloc(&a[j], m[j] * k[j] * batch_count[j] * sizeof(input_type_a)));
            HIP_CHECK(hipHostMalloc(&b[j], n[j] * k[j] * batch_count[j] * sizeof(input_type_b)));
            HIP_CHECK(hipHostMalloc(&c[j], m[j] * n[j] * batch_count[j] * sizeof(output_type)));
            HIP_CHECK(hipHostMalloc(&d[j], m[j] * n[j] * batch_count[j] * sizeof(output_type)));
            HIP_CHECK(hipHostMalloc(&alpha_vec[j], m[j] * batch_count[j] * sizeof(float)));

            for(int i = 0; i < m[j] * k[j] * batch_count[j]; i++)
                ((input_type_a*)a[j])[i] = static_cast<input_type_a>((rand() % 7) - 3);
            for(int i = 0; i < n[j] * k[j] * batch_count[j]; i++)
                ((input_type_b*)b[j])[i] = static_cast<input_type_b>((rand() % 7) - 3);
            for(int i = 0; i < m[j] * n[j] * batch_count[j]; i++)
                ((output_type*)c[j])[i] = static_cast<output_type>((rand() % 7) - 3);
            for(int i = 0; i < m[j] * batch_count[j]; i++)
                ((float*)alpha_vec[j])[i] = static_cast<float>((rand() % 7) - 3);
        }
        if(max_workspace_size > 0)
            HIP_CHECK(hipMalloc(&d_workspace, max_workspace_size));
    }

    ~runner_vec()
    {
        for(size_t j = 0; j < m.size(); j++)
        {
            HIP_CHECK(hipFree(a[j]));
            HIP_CHECK(hipFree(b[j]));
            HIP_CHECK(hipFree(c[j]));
            HIP_CHECK(hipFree(d[j]));
            HIP_CHECK(hipFree(alpha_vec[j]));
            HIP_CHECK(hipFree(d_a[j]));
            HIP_CHECK(hipFree(d_b[j]));
            HIP_CHECK(hipFree(d_c[j]));
            HIP_CHECK(hipFree(d_d[j]));
            HIP_CHECK(hipFree(d_alpha_vec[j]));
        }
        HIP_CHECK(hipFree(d_workspace));
        HIPBLASLT_CHECK(hipblasLtDestroy(handle));
        HIP_CHECK(hipStreamDestroy(stream));
    }

    void host_to_device()
    {
        for(size_t j = 0; j < m.size(); j++)
        {
            HIP_CHECK(hipMemcpyAsync(d_a[j],
                                     a[j],
                                     m[j] * k[j] * batch_count[j] * sizeof(input_type_a),
                                     hipMemcpyHostToDevice,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(d_b[j],
                                     b[j],
                                     n[j] * k[j] * batch_count[j] * sizeof(input_type_b),
                                     hipMemcpyHostToDevice,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(d_c[j],
                                     c[j],
                                     m[j] * n[j] * batch_count[j] * sizeof(output_type),
                                     hipMemcpyHostToDevice,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(d_alpha_vec[j],
                                     alpha_vec[j],
                                     m[j] * batch_count[j] * sizeof(float),
                                     hipMemcpyHostToDevice,
                                     stream));
        }
    }

    void device_to_host()
    {
        for(size_t j = 0; j < m.size(); j++)
        {
            HIP_CHECK(hipMemcpyAsync(d[j],
                                     d_d[j],
                                     m[j] * n[j] * batch_count[j] * sizeof(output_type),
                                     hipMemcpyDeviceToHost,
                                     stream));
        }
    }

    void run(const std::function<void()>& func)
    {
        host_to_device();

        static_cast<void>(func());

        device_to_host();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    std::vector<int64_t>    m;
    std::vector<int64_t>    n;
    std::vector<int64_t>    k;
    std::vector<int64_t>    batch_count;
    std::vector<alpha_type> alpha;
    std::vector<beta_type>  beta;

    std::vector<void*> a, b, c, d, alpha_vec; // host
    std::vector<void*> d_a, d_b, d_c, d_d, d_alpha_vec; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template<typename data_type>
struct layer_norm_runner
{
    layer_norm_runner(int64_t m, int64_t n) : m(m), n(n)
    {
        HIP_CHECK(hipStreamCreate(&stream));
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));

        HIP_CHECK(hipMalloc(&d_out, m * n * sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_mean, m * sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_invvar, m * sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_in, m * n * sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_gamma, n * sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_beta, n * sizeof(data_type)));

        HIP_CHECK(hipHostMalloc(&out, m * n * sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&mean, m * sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&invvar, m * sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&in, m * n * sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&gamma, n * sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&beta, n * sizeof(data_type)));

        for(int i = 0; i < m * n; i++)
            ((data_type*)in)[i] = static_cast<data_type>((rand() % 7) - 3);
        for(int i = 0; i < n; i++)
            ((data_type*)gamma)[i] = static_cast<data_type>((rand() % 7) - 3);
        for(int i = 0; i < n; i++)
            ((data_type*)beta)[i] = static_cast<data_type>((rand() % 7) - 3);
    }

    ~layer_norm_runner()
    {
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_mean));
        HIP_CHECK(hipFree(d_invvar));
        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_gamma));
        HIP_CHECK(hipFree(d_beta));

        HIP_CHECK(hipFree(out));
        HIP_CHECK(hipFree(mean));
        HIP_CHECK(hipFree(invvar));
        HIP_CHECK(hipFree(in));
        HIP_CHECK(hipFree(gamma));
        HIP_CHECK(hipFree(beta));

        HIPBLASLT_CHECK(hipblasLtDestroy(handle));
        HIP_CHECK(hipStreamDestroy(stream));
    }

    void host_to_device()
    {
        HIP_CHECK(
            hipMemcpyAsync(d_in, in, m * n * sizeof(data_type), hipMemcpyHostToDevice, stream));
        HIP_CHECK(
            hipMemcpyAsync(d_gamma, gamma, n * sizeof(data_type), hipMemcpyHostToDevice, stream));
        HIP_CHECK(
            hipMemcpyAsync(d_beta, beta, n * sizeof(data_type), hipMemcpyHostToDevice, stream));
    }

    void device_to_host()
    {
        HIP_CHECK(
            hipMemcpyAsync(out, d_out, m * n * sizeof(data_type), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(
            hipMemcpyAsync(mean, d_mean, m * sizeof(data_type), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(
            hipMemcpyAsync(invvar, d_invvar, m * sizeof(data_type), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        host_to_device();

        static_cast<void>(func());

        device_to_host();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t m;
    int64_t n;

    void *out, *mean, *invvar, *in, *gamma, *beta; // host
    void *d_out, *d_mean, *d_invvar, *d_in, *d_gamma, *d_beta; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template<typename data_type>
struct opt_amax_runner
{
    opt_amax_runner(int64_t m, int64_t n) : m(m), n(n)
    {
        HIP_CHECK(hipStreamCreate(&stream));
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));

        HIP_CHECK(hipMalloc(&d_out, sizeof(data_type)));
        HIP_CHECK(hipMalloc(&d_in, m * n * sizeof(data_type)));

        HIP_CHECK(hipHostMalloc(&out, sizeof(data_type)));
        HIP_CHECK(hipHostMalloc(&in, m * n * sizeof(data_type)));

        for(int i = 0; i < m * n; i++)
            ((data_type*)in)[i] = static_cast<data_type>((rand() % 7) - 3);
    }

    ~opt_amax_runner()
    {
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_in));

        HIP_CHECK(hipFree(out));
        HIP_CHECK(hipFree(in));

        HIPBLASLT_CHECK(hipblasLtDestroy(handle));
        HIP_CHECK(hipStreamDestroy(stream));
    }

    void host_to_device()
    {
        HIP_CHECK(
            hipMemcpyAsync(d_in, in, m * n * sizeof(data_type), hipMemcpyHostToDevice, stream));
    }

    void device_to_host()
    {
        HIP_CHECK(hipMemcpyAsync(out, d_out, sizeof(data_type), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        host_to_device();

        static_cast<void>(func());

        device_to_host();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t m;
    int64_t n;

    void *in, *out; // host
    void *d_in, *d_out; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

#endif // COMMON_HIPBLASLT_UTILS_HPP
