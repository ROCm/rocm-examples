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

#include "hiptensor_utils.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main()
{
    // 1. Check if F32 is supported.
    if(!is_f32_supported())
    {
        std::cout << "unsupported host device" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 2. Define type aliases.
    typedef float float_type_a;
    typedef float float_type_c;
    typedef float float_type_d;
    typedef float float_type_compute;

    // 3. Set up tensor data types.
    hiptensorDataType_t                type_a       = HIPTENSOR_R_32F;
    hiptensorDataType_t                type_c       = HIPTENSOR_R_32F;
    hiptensorDataType_t                type_d       = HIPTENSOR_R_32F;
    hiptensorComputeDescriptor_t const desc_compute = HIPTENSOR_COMPUTE_DESC_32F;

    // 4. Set scalar values.
    float_type_compute alpha = (float_type_compute)1.0f;
    float_type_compute gamma = (float_type_compute)2.0f;

    // 5. Define tensor operation.
    // D_{c,w,h} = alpha * A_{w,h,c}  + gamma * C_{w,h,c}

    // 6. Set up tensor modes.
    std::vector<int> mode_a{'w', 'h', 'c'};
    std::vector<int> mode_c{'w', 'h', 'c'};
    std::vector<int> mode_d{'c', 'w', 'h'};
    int              nmode_a = mode_a.size();
    int              nmode_c = mode_c.size();
    int              nmode_d = mode_d.size();

    // 7. Set up tensor extents.
    std::unordered_map<int, int64_t> extent;
    extent['h'] = 512;
    extent['w'] = 512;
    extent['c'] = 512;

    // 8. Calculate extent vectors.
    std::vector<int64_t> extent_a;
    for(auto mode : mode_a)
    {
        extent_a.push_back(extent[mode]);
    }
    std::vector<int64_t> extent_c;
    for(auto mode : mode_c)
    {
        extent_c.push_back(extent[mode]);
    }
    std::vector<int64_t> extent_d;
    for(auto mode : mode_d)
    {
        extent_d.push_back(extent[mode]);
    }

    // 9. Allocate device memory.
    size_t elements_a = 1;
    for(auto mode : mode_a)
    {
        elements_a *= extent[mode];
    }
    size_t elements_c = 1;
    for(auto mode : mode_c)
    {
        elements_c *= extent[mode];
    }
    size_t elements_d = 1;
    for(auto mode : mode_d)
    {
        elements_d *= extent[mode];
    }

    size_t size_a = sizeof(float_type_a) * elements_a;
    size_t size_c = sizeof(float_type_c) * elements_c;
    size_t size_d = sizeof(float_type_d) * elements_d;

    void *a_d, *c_d, *d_d;
    HIP_CHECK(hipMalloc((void**)&a_d, size_a));
    HIP_CHECK(hipMalloc((void**)&c_d, size_c));
    HIP_CHECK(hipMalloc((void**)&d_d, size_d));

    float_type_a* a;
    float_type_c* c;
    float_type_d* d;
    HIP_CHECK(hipHostMalloc((void**)&a, sizeof(float_type_a) * elements_a));
    HIP_CHECK(hipHostMalloc((void**)&c, sizeof(float_type_c) * elements_c));
    HIP_CHECK(hipHostMalloc((void**)&d, sizeof(float_type_d) * elements_d));

    // 10. Initialize data.
    for(size_t i = 0; i < elements_a; i++)
    {
        a[i] = (float)i;
        c[i] = static_cast<float>(i % 41);
    }

    HIP_CHECK(hipMemcpy(a_d, a, size_a, hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(c_d, c, size_c, hipMemcpyDefault));

    // 11. Initialize hipTensor.
    hiptensorHandle_t handle;
    HIPTENSOR_CHECK(hiptensorCreate(&handle));
    HIPTENSOR_CHECK(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    // 12. Create tensor descriptors.
    hiptensorTensorDescriptor_t desc_a = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &desc_a,
                                                    nmode_a,
                                                    extent_a.data(),
                                                    nullptr /* stride */,
                                                    type_a,
                                                    0));

    hiptensorTensorDescriptor_t desc_c = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &desc_c,
                                                    nmode_c,
                                                    extent_c.data(),
                                                    nullptr /* stride */,
                                                    type_c,
                                                    0));

    hiptensorTensorDescriptor_t desc_d = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &desc_d,
                                                    nmode_d,
                                                    extent_d.data(),
                                                    nullptr /* stride */,
                                                    type_d,
                                                    0));

    // 13. Create elementwise binary descriptor.
    hiptensorOperationDescriptor_t desc;
    HIPTENSOR_CHECK(hiptensorCreateElementwiseBinary(handle,
                                                     &desc,
                                                     desc_a,
                                                     mode_a.data(),
                                                     /* unary operator A  */ HIPTENSOR_OP_IDENTITY,
                                                     desc_c,
                                                     mode_c.data(),
                                                     /* unary operator C  */ HIPTENSOR_OP_IDENTITY,
                                                     desc_d,
                                                     mode_d.data(),
                                                     /* unary operator AC */ HIPTENSOR_OP_ADD,
                                                     desc_compute));

    // 14. Set algorithm.
    const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_CHECK(
        hiptensorCreatePlanPreference(handle, &plan_pref, algo, HIPTENSOR_JIT_MODE_NONE));

    // 15. Create plan.
    hiptensorPlan_t plan;
    HIPTENSOR_CHECK(
        hiptensorCreatePlan(handle, &plan, desc, plan_pref, 0 /* workspaceSizeLimit */));

    // 16. Run elementwise binary operation.
    HIPTENSOR_CHECK(hiptensorElementwiseBinaryExecute(handle,
                                                      plan,
                                                      (void*)&alpha,
                                                      a_d,
                                                      (void*)&gamma,
                                                      c_d,
                                                      d_d,
                                                      nullptr /* stream */));

#if !NDEBUG
    // 17. Print and store results.
    bool print_elements = false;
    bool store_elements = false;

    if(print_elements || store_elements)
    {
        HIP_CHECK(hipMemcpy(d, d_d, size_d, hipMemcpyDefault));
    }

    if(print_elements)
    {
        if(elements_a < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensor_print_array_elements(std::cout, a, elements_a);
            std::cout << std::endl;
        }

        if(elements_c < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensor_print_array_elements(std::cout, c, elements_c);
            std::cout << std::endl;
        }

        if(elements_d < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensor_print_array_elements(std::cout, d, elements_d);
            std::cout << std::endl;
        }
    }

    if(store_elements)
    {
        std::ofstream tensor_a, tensor_c, tensor_d;
        tensor_a.open("tensor_A.txt");
        hiptensor_print_elements_to_file(tensor_a, a, elements_a, ", ");
        tensor_a.close();

        tensor_c.open("tensor_C.txt");
        hiptensor_print_elements_to_file(tensor_c, c, elements_c, ", ");
        tensor_c.close();

        tensor_d.open("tensor_D_scale_contraction_results.txt");
        hiptensor_print_elements_to_file(tensor_d, d, elements_d, ", ");
        tensor_d.close();
    }
#endif

    // 18. Cleanup.
    HIPTENSOR_CHECK(hiptensorDestroy(handle));
    HIPTENSOR_CHECK(hiptensorDestroyPlan(plan));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(desc));
    HIPTENSOR_CHECK(hiptensorDestroyPlanPreference(plan_pref));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(desc_a));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(desc_c));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(desc_d));

    HIPTENSOR_FREE_HOST(a);
    HIPTENSOR_FREE_HOST(c);
    HIPTENSOR_FREE_HOST(d);
    HIPTENSOR_FREE_DEVICE(a_d);
    HIPTENSOR_FREE_DEVICE(c_d);
    HIPTENSOR_FREE_DEVICE(d_d);

    std::cout << "Finished!" << std::endl;
    return 0;
}
