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
        return 0;
    }

    // 2. Define type aliases.
    typedef float float_type_a;
    // typedef float float_type_b;
    typedef float float_type_c;
    typedef float float_type_compute;

    // 3. Set up tensor data types.
    hiptensorDataType_t type_a = HIPTENSOR_R_32F;
    hiptensorDataType_t type_c = HIPTENSOR_R_32F;
    // hiptensorComputeDescriptor_t       type_compute = HIPTENSOR_COMPUTE_DESC_32F;
    const hiptensorComputeDescriptor_t desc_compute = HIPTENSOR_COMPUTE_DESC_32F;

    // 4. Set scalar values.
    float_type_compute alpha = (float_type_compute)1.1f;
    float_type_compute beta  = (float_type_compute)0.f;

    // 5. Define tensor operation.
    // C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}

    // 6. Set up tensor modes.
    std::vector<int32_t> mode_a{'m', 'h', 'k', 'v'};
    std::vector<int32_t> mode_c{'k', 'v'};
    int32_t              nmode_a = mode_a.size();
    int32_t              nmode_c = mode_c.size();

    // 7. Set up tensor extents.
    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 3;
    extent['v'] = 4;
    extent['h'] = 5;
    extent['k'] = 6;

    // 8. Calculate extent vectors.
    std::vector<int64_t> extent_c;
    for(auto mode : mode_c)
    {
        extent_c.push_back(extent[mode]);
    }

    std::vector<int64_t> extent_a;
    for(auto mode : mode_a)
    {
        extent_a.push_back(extent[mode]);
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

    size_t size_a = sizeof(float_type_a) * elements_a;
    size_t size_c = sizeof(float_type_c) * elements_c;

    void *a_d, *c_d;
    HIP_CHECK(hipMalloc((void**)&a_d, size_a));
    HIP_CHECK(hipMalloc((void**)&c_d, size_c));

    float_type_a *a, *c;
    HIP_CHECK(hipHostMalloc((void**)&a, sizeof(float_type_a) * elements_a));
    HIP_CHECK(hipHostMalloc((void**)&c, sizeof(float_type_c) * elements_c));

    // 10. Initialize data.
    for(size_t i = 0; i < elements_a; i++)
    {
        a[i] = (float)i;
    }
    for(size_t i = 0; i < elements_c; i++)
    {
        c[i] = (float)(i & 1);
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
                                                    NULL /* stride */,
                                                    type_a,
                                                    0));

    hiptensorTensorDescriptor_t desc_c = nullptr;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle,
                                                    &desc_c,
                                                    nmode_c,
                                                    extent_c.data(),
                                                    NULL /* stride */,
                                                    type_c,
                                                    0));

    // 13. Set up reduction operation.
    const hiptensorOperator_t op_reduce = HIPTENSOR_OP_ADD;

    // 14. Create reduction descriptor.
    hiptensorOperationDescriptor_t desc;
    HIPTENSOR_CHECK(hiptensorCreateReduction(handle,
                                             &desc,
                                             desc_a,
                                             mode_a.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             desc_c,
                                             mode_c.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             desc_c,
                                             mode_c.data(),
                                             op_reduce,
                                             desc_compute));

    // 15. Set algorithm.
    const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_CHECK(
        hiptensorCreatePlanPreference(handle, &plan_pref, algo, HIPTENSOR_JIT_MODE_NONE));

    // 16. Query workspace estimate.
    uint64_t                            worksize       = 0;
    const hiptensorWorksizePreference_t workspace_pref = HIPTENSOR_WORKSPACE_DEFAULT;
    HIPTENSOR_CHECK(
        hiptensorEstimateWorkspaceSize(handle, desc, plan_pref, workspace_pref, &worksize));
    void* work = nullptr;
    if(worksize > 0)
    {
        if(hipSuccess != hipMalloc(&work, worksize))
        {
            work     = nullptr;
            worksize = 0;
        }
    }

    // 17. Create plan.
    hiptensorPlan_t plan;
    HIPTENSOR_CHECK(hiptensorCreatePlan(handle, &plan, desc, plan_pref, worksize));

    // 18. Run reduction.
    HIPTENSOR_CHECK(hiptensorReduce(handle,
                                    plan,
                                    (const void*)&alpha,
                                    a_d,
                                    (const void*)&beta,
                                    c_d,
                                    c_d,
                                    work,
                                    worksize,
                                    0));

#if !NDEBUG
    // 19. Print and store results.
    bool print_elements = true;
    bool store_elements = false;

    if(print_elements || store_elements)
    {
        HIP_CHECK(hipMemcpy(c, c_d, size_c, hipMemcpyDefault));
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
    }

    if(store_elements)
    {
        std::ofstream tensor_a, tensor_c;
        tensor_a.open("tensor_A.txt");
        hiptensor_print_elements_to_file(tensor_a, a, elements_a, ", ");
        tensor_a.close();

        tensor_c.open("tensor_C_scale_contraction_results.txt");
        hiptensor_print_elements_to_file(tensor_c, c, elements_c, ", ");
        tensor_c.close();
    }
#endif

    // 20. Cleanup.
    HIPTENSOR_CHECK(hiptensorDestroy(handle));
    HIPTENSOR_CHECK(hiptensorDestroyPlan(plan));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(desc));
    HIPTENSOR_CHECK(hiptensorDestroyPlanPreference(plan_pref));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(desc_a));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(desc_c));

    HIPTENSOR_FREE_HOST(a);
    HIPTENSOR_FREE_HOST(c);
    HIPTENSOR_FREE_DEVICE(a_d);
    HIPTENSOR_FREE_DEVICE(c_d);
    HIPTENSOR_FREE_DEVICE(work);

    std::cout << "Finished!" << std::endl;
    return 0;
}
