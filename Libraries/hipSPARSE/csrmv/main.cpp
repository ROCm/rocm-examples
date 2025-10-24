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
#include "hipsparse_utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char* argv[])
{
    // Parse user inputs
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("n", "ndim", 5, "Problem dimension");
    parser.set_optional<int>("t", "trials", 200, "Number of trials");
    parser.set_optional<int>("b", "batch_size", 1, "Batch size");
    parser.run_and_exit_if_error();

    int ndim       = parser.get<int>("n");
    int trials     = parser.get<int>("t");
    int batch_size = parser.get<int>("b");

    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipDeviceProp_t dev_prop;
    int             device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));
    printf("Device: %s\n", dev_prop.name);

    // Generate problem
    std::vector<int>    h_aptr;
    std::vector<int>    h_acol;
    std::vector<double> h_aval;
    int m   = gen_2d_laplacian(ndim, h_aptr, h_acol, h_aval, HIPSPARSE_INDEX_BASE_ZERO);
    int n   = m;
    int nnz = h_aptr[m];

    // Sample some random data
    srand(12345ULL);

    double h_alpha = static_cast<double>(rand()) / RAND_MAX;
    double h_beta  = 0.0;

    std::vector<double> h_x(n);
    hipsparseInit(h_x, 1, n);

    // Matrix descriptor
    hipsparseMatDescr_t descr_a;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr_a));

    // Offload data to device
    int*    d_aptr = NULL;
    int*    d_acol = NULL;
    double* d_aval = NULL;
    double* d_x    = NULL;
    double* d_y    = NULL;

    HIP_CHECK(hipMalloc((void**)&d_aptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_acol, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&d_aval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&d_x, sizeof(double) * n));
    HIP_CHECK(hipMalloc((void**)&d_y, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(d_aptr, h_aptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_acol, h_acol.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aval, h_aval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(double) * n, hipMemcpyHostToDevice));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call hipsparse csrmv
        HIPSPARSE_CHECK(hipsparseDcsrmv(handle,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        m,
                                        n,
                                        nnz,
                                        &h_alpha,
                                        descr_a,
                                        d_aval,
                                        d_aptr,
                                        d_acol,
                                        d_x,
                                        &h_beta,
                                        d_y));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call hipsparse csrmv
            HIPSPARSE_CHECK(hipsparseDcsrmv(handle,
                                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                            m,
                                            n,
                                            nnz,
                                            &h_alpha,
                                            descr_a,
                                            d_aval,
                                            d_aptr,
                                            d_acol,
                                            d_x,
                                            &h_beta,
                                            d_y));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth
        = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz)) / time
          / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;
    printf("m\t\tn\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tusec\n");
    printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
           m,
           n,
           nnz,
           h_alpha,
           h_beta,
           gflops,
           bandwidth,
           time);

    // Clear up on device
    HIP_CHECK(hipFree(d_aptr));
    HIP_CHECK(hipFree(d_acol));
    HIP_CHECK(hipFree(d_aval));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_a));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
