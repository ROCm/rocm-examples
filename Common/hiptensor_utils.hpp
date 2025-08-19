// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_HIPTENSOR_UTILS_HPP
#define COMMON_HIPTENSOR_UTILS_HPP

#include "example_utils.hpp"

#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <hiptensor/internal/types.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

/// \brief Converts a \p hiptensorStatus_t variable to its correspondent string.
inline const char* hiptensor_status_to_string(hiptensorStatus_t status)
{
    switch(status)
    {
        case HIPTENSOR_STATUS_SUCCESS: return "HIPTENSOR_STATUS_SUCCESS";
        case HIPTENSOR_STATUS_NOT_INITIALIZED: return "HIPTENSOR_STATUS_NOT_INITIALIZED";
        case HIPTENSOR_STATUS_ALLOC_FAILED: return "HIPTENSOR_STATUS_ALLOC_FAILED";
        case HIPTENSOR_STATUS_INVALID_VALUE: return "HIPTENSOR_STATUS_INVALID_VALUE";
        case HIPTENSOR_STATUS_ARCH_MISMATCH: return "HIPTENSOR_STATUS_ARCH_MISMATCH";
        case HIPTENSOR_STATUS_EXECUTION_FAILED: return "HIPTENSOR_STATUS_EXECUTION_FAILED";
        case HIPTENSOR_STATUS_INTERNAL_ERROR: return "HIPTENSOR_STATUS_INTERNAL_ERROR";
        case HIPTENSOR_STATUS_NOT_SUPPORTED: return "HIPTENSOR_STATUS_NOT_SUPPORTED";
        case HIPTENSOR_STATUS_CK_ERROR: return "HIPTENSOR_STATUS_CK_ERROR";
        case HIPTENSOR_STATUS_HIP_ERROR: return "HIPTENSOR_STATUS_HIP_ERROR";
        case HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
            return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
        case HIPTENSOR_STATUS_INSUFFICIENT_DRIVER: return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
        case HIPTENSOR_STATUS_IO_ERROR: return "HIPTENSOR_STATUS_IO_ERROR";
        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something hiptensor would provide
        default: return "<unknown hiptensorStatus_t value>";
    }
}

/// \brief Checks if the provided status code is \p HIPTENSOR_STATUS_SUCCESS and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIPTENSOR_CHECK(condition)                                                               \
    {                                                                                            \
        const hiptensorStatus_t status = (condition);                                            \
        if(status != HIPTENSOR_STATUS_SUCCESS)                                                   \
        {                                                                                        \
            std::cerr << "hipTensor error encountered: \"" << hiptensor_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;                   \
            std::exit(error_exit_code);                                                          \
        }                                                                                        \
    }

#define MAX_ELEMENTS_PRINT_COUNT 512

#define HIPTENSOR_FREE_DEVICE(ptr) \
    if(ptr != nullptr)             \
    {                              \
        HIP_CHECK(hipFree(ptr));   \
    }

#define HIPTENSOR_FREE_HOST(ptr)     \
    if(ptr != nullptr)               \
    {                                \
        HIP_CHECK(hipHostFree(ptr)); \
    }

inline bool is_f32_supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    HIP_CHECK(hipGetDevice(&mHandle));
    HIP_CHECK(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx908") != std::string::npos)
           || (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos)
           || (deviceName.find("gfx950") != std::string::npos);
}

inline bool is_f64_supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    HIP_CHECK(hipGetDevice(&mHandle));
    HIP_CHECK(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos)
           || (deviceName.find("gfx950") != std::string::npos);
}

template<typename T>
void hiptensor_print_array_elements(std::ostream& stream, T* vec, size_t size)
{
    for(size_t index = 0; index < size; ++index)
    {
        if constexpr(std::is_same_v<T, float2> || std::is_same_v<T, double2>)
        {
            stream << "(" << vec[index].x << ", " << vec[index].y << ")";
        }
        else if constexpr(std::is_same_v<T, _Float16>)
        {
            stream << static_cast<float>(vec[index]);
        }
        else
        {
            stream << vec[index];
        }

        if(index != size - 1)
        {
            stream << ", ";
        }
    }
}

template<typename S>
void hiptensor_print_vector_elements(const std::vector<S>& vec, std::string sep = " ")
{
    for(auto& elem : vec)
    {
        std::cout << elem;
        if(&elem != &vec.back())
        {
            std::cout << sep;
        }
    }
}

template<typename F>
void hiptensor_print_elements_to_file(std::ofstream& fs,
                                      F*             output,
                                      size_t         size,
                                      std::string    sep = " ")
{
    if(!fs.is_open())
    {
        std::cout << "File not found!\n";
        return;
    }

    for(size_t i = 0; i < size; i++)
    {
        if constexpr(std::is_same_v<F, float2> || std::is_same_v<F, double2>)
        {
            fs << "(" << output[i].x << ", " << output[i].y << ")";
        }
        else if constexpr(std::is_same_v<F, _Float16>)
        {
            fs << static_cast<float>(output[i]);
        }
        else
        {
            fs << static_cast<F>(output[i]);
        }

        if(i != size - 1)
        {
            fs << sep;
        }
    }
    return;
}

// Bilinear contraction sample function
template<typename ADataType,
         typename BDataType,
         typename CDataType,
         hiptensorDataType_t          typeA,
         hiptensorDataType_t          typeB,
         hiptensorDataType_t          typeC,
         hiptensorComputeDescriptor_t typeCompute>
int bilinear_contraction_sample(void* alpha, void* beta)
{
    // Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta * C_{m,n,u,v}

    std::vector<int> mode_c{'m', 'n', 'u', 'v'};
    std::vector<int> mode_a{'m', 'n', 'h', 'k'};
    std::vector<int> mode_b{'u', 'v', 'h', 'k'};

    int nmode_a = mode_a.size();
    int nmode_b = mode_b.size();
    int nmode_c = mode_c.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 4;
    extent['n'] = 3;
    extent['u'] = 4;
    extent['v'] = 3;
    extent['h'] = 6;
    extent['k'] = 5;

    std::vector<int64_t> c_ms_ns_lengths;
    for(auto mode : mode_c)
    {
        c_ms_ns_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : mode_a)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> b_ns_ks_lengths;
    for(auto mode : mode_b)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
    }

    // Allocating data
    std::cout << "Initializing host data..." << std::endl;

    size_t elements_a = std::accumulate(a_ms_ks_lengths.begin(),
                                        a_ms_ks_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());
    size_t elements_b = std::accumulate(b_ns_ks_lengths.begin(),
                                        b_ns_ks_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());
    size_t elements_c = std::accumulate(c_ms_ns_lengths.begin(),
                                        c_ms_ns_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());

    size_t size_a = sizeof(ADataType) * elements_a;
    size_t size_b = sizeof(BDataType) * elements_b;
    size_t size_c = sizeof(CDataType) * elements_c;

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    CDataType* C = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&A, size_a));
    HIP_CHECK(hipHostMalloc((void**)&B, size_b));
    HIP_CHECK(hipHostMalloc((void**)&C, size_c));

    void *A_d, *B_d, *C_d;

    HIP_CHECK(hipMalloc(static_cast<void**>(&A_d), size_a));
    HIP_CHECK(hipMalloc(static_cast<void**>(&B_d), size_b));
    HIP_CHECK(hipMalloc(static_cast<void**>(&C_d), size_c));

    // Initialize data
    int init_method = 1; // TODO read value from commandline
    for(size_t i = 0; i < elements_a; i++)
    {
        if(init_method == 0)
        {
            A[i] = ADataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            A[i] = (ADataType)(float(i) / 100);
        }
    }

    for(size_t i = 0; i < elements_b; i++)
    {
        if(init_method == 0)
        {
            B[i] = BDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            B[i] = (BDataType)(float(i) / 100);
        }
    }

    for(size_t i = 0; i < elements_c; i++)
    {
        if(init_method == 0)
        {
            C[i] = CDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            C[i] = (BDataType)(float(i) / 100);
        }
    }

    // Transfer the Host Tensor to Device Memory
    std::cout << "Initializing device data..." << std::endl;

    HIP_CHECK(hipMemcpy(A_d, static_cast<const void*>(A), size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, static_cast<const void*>(B), size_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_d, static_cast<const void*>(C), size_c, hipMemcpyHostToDevice));

    // Retrieve the memory alignment for each tensor
    uint32_t          alignment_requirement = 1;
    hiptensorHandle_t handle;
    HIPTENSOR_CHECK(hiptensorCreate(&handle));

    HIPTENSOR_CHECK(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    // Initialize tensors with the input lengths
    hiptensorTensorDescriptor_t a_ms_ks = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &a_ms_ks,
                                                    nmode_a,
                                                    a_ms_ks_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeA,
                                                    alignment_requirement));

    hiptensorTensorDescriptor_t b_ns_ks = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &b_ns_ks,
                                                    nmode_b,
                                                    b_ns_ks_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeB,
                                                    alignment_requirement));

    hiptensorTensorDescriptor_t c_ms_ns = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &c_ms_ns,
                                                    nmode_c,
                                                    c_ms_ns_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeC,
                                                    alignment_requirement));

    // Create Contraction Descriptor
    hiptensorOperationDescriptor_t desc;
    HIPTENSOR_CHECK(hiptensorCreateContraction(handle,
                                               &desc,
                                               a_ms_ks,
                                               mode_a.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               b_ns_ks,
                                               mode_b.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               c_ms_ns,
                                               mode_c.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               c_ms_ns,
                                               mode_c.data(),
                                               typeCompute));

    // Set the algorithm to use
    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_CHECK(hiptensorCreatePlanPreference(handle,
                                                  &plan_pref,
                                                  HIPTENSOR_ALGO_ACTOR_CRITIC,
                                                  HIPTENSOR_JIT_MODE_NONE));

    // Query workspace
    uint64_t worksize = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(handle,
                                                   desc,
                                                   plan_pref,
                                                   HIPTENSOR_WORKSPACE_DEFAULT,
                                                   &worksize));

    // Create Contraction Plan
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorPlan_t plan;
    HIPTENSOR_CHECK(hiptensorCreatePlan(handle, &plan, desc, plan_pref, worksize));

    // TODO query actually used workspace
    void* workspace = nullptr;

    if(worksize > 0)
    {
        HIP_CHECK(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    std::cout << "Launching contraction kernel..." << std::endl;

    HIPTENSOR_CHECK(hiptensorContract(handle,
                                      plan,
                                      alpha,
                                      A_d,
                                      B_d,
                                      beta,
                                      C_d,
                                      C_d,
                                      workspace,
                                      worksize,
                                      0 /* stream */));

#if !NDEBUG
    bool print_elements = false;
    bool store_elements = false;

    if(print_elements)
    {
        if(elements_a < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensor_print_array_elements(std::cout, A, elements_a);
            std::cout << std::endl;
        }

        if(elements_b < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensor_print_array_elements(std::cout, B, elements_b);
            std::cout << std::endl;
        }

        if(elements_c < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensor_print_array_elements(std::cout, C, elements_c);
            std::cout << std::endl;
        }

        HIP_CHECK(hipMemcpy(C, C_d, size_c, hipMemcpyDeviceToHost));

        if(elements_c < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensor_print_array_elements(std::cout, C, elements_c);
            std::cout << std::endl;
        }
    }

    if(store_elements)
    {
        std::ofstream tensor_a, tensor_b, tensor_c;
        tensor_a.open("tensor_A.txt");
        hiptensor_print_elements_to_file(tensor_a, A, elements_a, ", ");
        tensor_a.close();

        tensor_b.open("tensor_B.txt");
        hiptensor_print_elements_to_file(tensor_b, B, elements_b, ", ");
        tensor_b.close();

        tensor_c.open("tensor_C.txt");
        hiptensor_print_elements_to_file(tensor_c, C, elements_c, ", ");
        tensor_c.close();

        HIP_CHECK(hipMemcpy(C, C_d, size_c, hipMemcpyDeviceToHost));

        tensor_c.open("tensor_C_scale_contraction_results.txt");
        hiptensor_print_elements_to_file(tensor_c, C, elements_c, ", ");
        tensor_c.close();
    }
#endif

    HIPTENSOR_CHECK(hiptensorDestroy(handle));
    HIPTENSOR_CHECK(hiptensorDestroyPlanPreference(plan_pref));
    HIPTENSOR_CHECK(hiptensorDestroyPlan(plan));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        hiptensorDestroyTensorDescriptor(a_ms_ks);
        a_ms_ks = nullptr;
    }
    if(b_ns_ks)
    {
        hiptensorDestroyTensorDescriptor(b_ns_ks);
        b_ns_ks = nullptr;
    }
    if(c_ms_ns)
    {
        hiptensorDestroyTensorDescriptor(c_ms_ns);
        c_ms_ns = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}

// Scale contraction sample function
template<typename ADataType,
         typename BDataType,
         typename DDataType,
         hiptensorDataType_t          typeA,
         hiptensorDataType_t          typeB,
         hiptensorDataType_t          typeD,
         hiptensorComputeDescriptor_t typeCompute>
int scale_contraction_sample(void* alpha)
{
    // Computing: C_{m,n,u,v} = A_{m,n,h,k} B_{h,k,u,v}

    std::vector<int> mode_d{'m', 'n', 'u', 'v'};
    std::vector<int> mode_a{'m', 'n', 'h', 'k'};
    std::vector<int> mode_b{'u', 'v', 'h', 'k'};

    int nmode_a = mode_a.size();
    int nmode_b = mode_b.size();
    int nmode_d = mode_d.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 4;
    extent['n'] = 3;
    extent['u'] = 4;
    extent['v'] = 3;
    extent['h'] = 6;
    extent['k'] = 5;

    std::vector<int64_t> d_ms_ns_lengths;
    for(auto mode : mode_d)
    {
        d_ms_ns_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : mode_a)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> b_ns_ks_lengths;
    for(auto mode : mode_b)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
    }

    // Allocating data
    std::cout << "Initializing host data..." << std::endl;

    size_t elements_a = std::accumulate(a_ms_ks_lengths.begin(),
                                        a_ms_ks_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());
    size_t elements_b = std::accumulate(b_ns_ks_lengths.begin(),
                                        b_ns_ks_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());
    size_t elements_d = std::accumulate(d_ms_ns_lengths.begin(),
                                        d_ms_ns_lengths.end(),
                                        size_t{1},
                                        std::multiplies<size_t>());

    size_t size_a = sizeof(ADataType) * elements_a;
    size_t size_b = sizeof(BDataType) * elements_b;
    size_t size_d = sizeof(DDataType) * elements_d;

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    DDataType* D = nullptr;
    HIP_CHECK(hipHostMalloc((void**)&A, size_a));
    HIP_CHECK(hipHostMalloc((void**)&B, size_b));
    HIP_CHECK(hipHostMalloc((void**)&D, size_d));

    void *A_d, *B_d, *D_d;

    HIP_CHECK(hipMalloc(static_cast<void**>(&A_d), size_a));
    HIP_CHECK(hipMalloc(static_cast<void**>(&B_d), size_b));
    HIP_CHECK(hipMalloc(static_cast<void**>(&D_d), size_d));

    // Initialize data
    int init_method = 1; // TODO read the value from command line
    for(size_t i = 0; i < elements_a; i++)
    {
        if(init_method == 0)
        {
            A[i] = ADataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            A[i] = (ADataType)(float(i) / 100);
        }
    }

    for(size_t i = 0; i < elements_b; i++)
    {
        if(init_method == 0)
        {
            B[i] = BDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            B[i] = (BDataType)(float(i) / 100);
        }
    }

    for(size_t i = 0; i < elements_d; i++)
    {
        D[i] = std::numeric_limits<DDataType>::signaling_NaN();
    }

    // Transfer the Host Tensor to Device Memory
    std::cout << "Initializing device data..." << std::endl;

    HIP_CHECK(hipMemcpy(A_d, static_cast<const void*>(A), size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, static_cast<const void*>(B), size_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(D_d, 0, size_d));

    // Retrieve the memory alignment for each tensor
    uint32_t          alignment_requirement = 1;
    hiptensorHandle_t handle;
    HIPTENSOR_CHECK(hiptensorCreate(&handle));

    HIPTENSOR_CHECK(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    // Initialize tensors with the input lengths
    hiptensorTensorDescriptor_t a_ms_ks;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &a_ms_ks,
                                                    nmode_a,
                                                    a_ms_ks_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeA,
                                                    alignment_requirement));

    hiptensorTensorDescriptor_t b_ns_ks;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &b_ns_ks,
                                                    nmode_b,
                                                    b_ns_ks_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeB,
                                                    alignment_requirement));

    hiptensorTensorDescriptor_t d_ms_ns;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &d_ms_ns,
                                                    nmode_d,
                                                    d_ms_ns_lengths.data(),
                                                    NULL, /*stride*/
                                                    typeD,
                                                    alignment_requirement));

    // Create Contraction Descriptor
    hiptensorOperationDescriptor_t desc;
    HIPTENSOR_CHECK(hiptensorCreateContraction(handle,
                                               &desc,
                                               a_ms_ks,
                                               mode_a.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               b_ns_ks,
                                               mode_b.data(),
                                               HIPTENSOR_OP_IDENTITY,
                                               nullptr,
                                               nullptr,
                                               HIPTENSOR_OP_IDENTITY,
                                               d_ms_ns,
                                               mode_d.data(),
                                               typeCompute));

    // Set the algorithm to use
    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_CHECK(hiptensorCreatePlanPreference(handle,
                                                  &plan_pref,
                                                  HIPTENSOR_ALGO_ACTOR_CRITIC,
                                                  HIPTENSOR_JIT_MODE_NONE));

    // Query workspace
    uint64_t worksize = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(handle,
                                                   desc,
                                                   plan_pref,
                                                   HIPTENSOR_WORKSPACE_DEFAULT,
                                                   &worksize));

    // Create Contraction Plan
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorPlan_t plan;
    HIPTENSOR_CHECK(hiptensorCreatePlan(handle, &plan, desc, plan_pref, worksize));

    // TODO query actually used workspace
    void* workspace = nullptr;

    if(worksize > 0)
    {
        HIP_CHECK(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    std::cout << "Launching contraction kernel..." << std::endl;

    HIPTENSOR_CHECK(hiptensorContract(handle,
                                      plan,
                                      alpha,
                                      A_d,
                                      B_d,
                                      nullptr,
                                      nullptr,
                                      D_d,
                                      workspace,
                                      worksize,
                                      0 /* stream */));

#if !NDEBUG
    bool print_elements = false;
    bool store_elements = false;

    if(print_elements || store_elements)
    {
        HIP_CHECK(hipMemcpy(D, D_d, size_d, hipMemcpyDeviceToHost));
    }

    if(print_elements)
    {
        if(elements_a < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensor_print_array_elements(std::cout, A, elements_a);
            std::cout << std::endl;
        }

        if(elements_b < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensor_print_array_elements(std::cout, B, elements_b);
            std::cout << std::endl;
        }

        if(elements_d < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensor_print_array_elements(std::cout, D, elements_d);
            std::cout << std::endl;
        }
    }

    if(store_elements)
    {
        std::ofstream tensor_a, tensor_b, tensor_d;
        tensor_a.open("tensor_A.txt");
        hiptensor_print_elements_to_file(tensor_a, A, elements_a, ", ");
        tensor_a.close();

        tensor_b.open("tensor_B.txt");
        hiptensor_print_elements_to_file(tensor_b, B, elements_b, ", ");
        tensor_b.close();

        tensor_d.open("tensor_D_scale_contraction_results.txt");
        hiptensor_print_elements_to_file(tensor_d, D, elements_d, ", ");
        tensor_d.close();
    }
#endif

    HIPTENSOR_CHECK(hiptensorDestroy(handle));
    HIPTENSOR_CHECK(hiptensorDestroyPlan(plan));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(a_ms_ks));
        a_ms_ks = nullptr;
    }
    if(b_ns_ks)
    {
        HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(b_ns_ks));
        b_ns_ks = nullptr;
    }
    if(d_ms_ns)
    {
        HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(d_ms_ns));
        d_ms_ns = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(D);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(D_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}

#endif // COMMON_HIPTENSOR_UTILS_HPP
