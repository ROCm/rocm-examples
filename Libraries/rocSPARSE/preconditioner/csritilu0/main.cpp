// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"
#include "rocsparse_utils.hpp"

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main()
{
    // 1. Set up input data.
    //
    // A = (  1  0  0  0  0  0 ) * (  2  3  4  5  6  7 )
    //     (  2  1  0  0  0  0 )   (  0  2  3  4  5  6 )
    //     (  3  2  1  0  0  0 )   (  0  0  2  3  4  5 )
    //     (  4  3  2  1  0  0 )   (  0  0  0  2  3  4 )
    //     (  5  4  3  2  1  0 )   (  0  0  0  0  2  3 )
    //     (  6  5  4  3  2  1 )   (  0  0  0  0  0  2 )
    //
    //   = (  2   3   4   5   6   7   )
    //     (  4   8   11  14  17  20  )
    //     (  6   13  20  26  32  38  )
    //     (  8   18  29  40  50  60  )
    //     (  10  23  38  54  70  85  )
    //     (  12  28  47  68  90  112 )

    // Number of rows and columns of the input matrix.
    constexpr rocsparse_int n = 6;

    // Number of non-zero elements.
    constexpr rocsparse_int nnz = 36;

    // CSR values vector.
    // clang-format off
    constexpr std::array<double, nnz> h_csr_val
        = {2,   3,   4,   5,   6,   7,
           4,   8,   11,  14,  17,  20,
           6,   13,  20,  26,  32,  38,
           8,   18,  29,  40,  50,  60,
           10,  23,  38,  54,  70,  85,
           12,  28,  47,  68,  90,  112};
    // clang-format on

    // CSR row pointers vector.
    constexpr std::array<rocsparse_int, n + 1> h_csr_row_ptr = {0, 6, 12, 18, 24, 30, 36};

    // CSR column indices vector.
    constexpr std::array<rocsparse_int, nnz> h_csr_col_ind
        = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
           0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};

    // 2. Allocate device memory and offload input data to the device.
    rocsparse_int* d_csr_row_ptr{};
    rocsparse_int* d_csr_col_ind{};
    double*        d_csr_val{};
    double*        d_LU{};

    constexpr size_t csr_row_ptr_size = sizeof(*d_csr_row_ptr) * (n + 1);
    constexpr size_t csr_col_ind_size = sizeof(*d_csr_col_ind) * nnz;
    constexpr size_t csr_val_size     = sizeof(*d_csr_val) * nnz;
    constexpr size_t length_LU        = n * n;
    constexpr size_t LU_size          = sizeof(*d_LU) * length_LU;
    rocsparse_int    max_iter         = 100; /*maximum number of iterations*/
    rocsparse_int    free_iter        = 20;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, csr_row_ptr_size));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, csr_col_ind_size));
    HIP_CHECK(hipMalloc(&d_csr_val, csr_val_size));
    HIP_CHECK(hipMalloc(&d_LU, LU_size));

    HIP_CHECK(
        hipMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), csr_row_ptr_size, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_csr_col_ind, h_csr_col_ind.data(), csr_col_ind_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val.data(), csr_val_size, hipMemcpyHostToDevice));

    // 3. Initialize rocSPARSE and prepare utility variables for csritilu0 invocation.
    // Initialize rocSPARSE by creating a handle.
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Matrix descriptor.
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Iterative ILU0 (incomplete LU factorization with 0 fill-ins) algorithm used.
    constexpr rocsparse_itilu0_alg alg = rocsparse_itilu0_alg_default;

    // Options selected for the iterative algorithm:
    constexpr rocsparse_int option = rocsparse_itilu0_option_stopping_criteria
                                     | rocsparse_itilu0_option_compute_nrm_correction
                                     | rocsparse_itilu0_option_compute_nrm_residual
                                     | rocsparse_itilu0_option_convergence_history;

    // Tolerance.
    double tol = 1.0e5 * std::numeric_limits<double>::epsilon(); /*2.22045e-11*/

    // Indexing: zero based.
    constexpr rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // Data type: 64 bit floating point real.
    constexpr rocsparse_datatype data_type = rocsparse_datatype_f64_r;

    // 4. Sort CSR matrix before calling any of the iterative-ILU0-related functions.
    rocsparse_int* perm;
    HIP_CHECK(hipMalloc(&perm, csr_col_ind_size));
    ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, perm));

    // Query the required buffer size in bytes and allocate a temporary buffer for sorting.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t sort_buffer_size;
    void*  sort_temp_buffer{};
    ROCSPARSE_CHECK(rocsparse_csrsort_buffer_size(handle,
                                                  n,
                                                  n,
                                                  nnz,
                                                  d_csr_row_ptr,
                                                  d_csr_col_ind,
                                                  &sort_buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    HIP_CHECK(hipMalloc(&sort_temp_buffer, sort_buffer_size));

    // Sort the CSR matrix applying the identity permutation.
    ROCSPARSE_CHECK(rocsparse_csrsort(handle,
                                      n,
                                      n,
                                      nnz,
                                      descr,
                                      d_csr_row_ptr,
                                      d_csr_col_ind,
                                      perm,
                                      sort_temp_buffer));

    // Gather sorted values array.
    ROCSPARSE_CHECK(rocsparse_dgthr(handle, nnz, d_csr_val, d_csr_val, perm, idx_base));

    // 5. Query the required buffer size in bytes for the iterative-ILU0-related functions.
    // This function is non blocking and executed asynchronously with respect to the host.
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_csritilu0_buffer_size(handle,
                                                    alg,
                                                    option,
                                                    max_iter,
                                                    n,
                                                    nnz,
                                                    d_csr_row_ptr,
                                                    d_csr_col_ind,
                                                    idx_base,
                                                    data_type,
                                                    &buffer_size));
    // No synchronization with the device is needed because for scalar results, when using host
    // pointer mode (the default pointer mode) this function blocks the CPU till the GPU has copied
    // the results back to the host. See rocsparse_set_pointer_mode.

    // Allocate temporary buffer.
    void* temp_buffer{};
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // 6. Perform the preprocessing.
    ROCSPARSE_CHECK(rocsparse_csritilu0_preprocess(handle,
                                                   alg,
                                                   option,
                                                   max_iter,
                                                   n,
                                                   nnz,
                                                   d_csr_row_ptr,
                                                   d_csr_col_ind,
                                                   idx_base,
                                                   data_type,
                                                   buffer_size,
                                                   temp_buffer));

    // 7. Perform the iterative incomplete LU factorization.
    HIP_CHECK(hipMemset(d_LU, 0, LU_size));
    ROCSPARSE_CHECK(rocsparse_dcsritilu0_compute_ex(handle,
                                                    alg,
                                                    option,
                                                    &max_iter,
                                                    free_iter,
                                                    tol,
                                                    n,
                                                    nnz,
                                                    d_csr_row_ptr,
                                                    d_csr_col_ind,
                                                    d_csr_val,
                                                    d_LU,
                                                    idx_base,
                                                    buffer_size,
                                                    temp_buffer));

    // 8. Fetch the convergence data.
    std::cout << "Iterations performed: " << max_iter << std::endl;

    // The history vector has space for the corrections and the residual of each iteration (whether
    // they are computed or not).
    // Therefore, the size of the data collected is twice the number of iterations needed.
    std::vector<double> history(2 * max_iter);
    ROCSPARSE_CHECK(
        rocsparse_dcsritilu0_history(handle, alg, &max_iter, history.data(), buffer_size, temp_buffer));

    const double last_residual       = history[2 * max_iter - 1];
    const bool   csritilu0_converges = last_residual <= tol;

    int errors{};

    if(!csritilu0_converges)
    {
        std::cout << "Residual in last iteration " << last_residual
                  << " is greater or equal than tolerance " << tol
                  << ". Iterative incomplete LU factorization was unsuccessful." << std::endl;
        errors++;
    }
    else
    {
        // 9. Check errors and print the resulting matrices.
        std::vector<double> LU(length_LU);
        HIP_CHECK(hipMemcpy(LU.data(), d_LU, LU_size, hipMemcpyDeviceToHost));

        // Expected L and U matrices in dense format (row major).
        constexpr std::array<double, length_LU> L_expected
            = {1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0,
               4, 3, 2, 1, 0, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1};
        constexpr std::array<double, length_LU> U_expected
            = {2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 0, 0, 2, 3, 4, 5,
               0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 2};
        std::cout
            << "Iterative incomplete LU factorization A ~= L * U successfully computed with L "
               "matrix: \n";

        // L matrix is stored in the lower part of A. The diagonal is not stored as it is known
        // that all the diagonal elements are the multiplicative identity (1 in this case).
        for(rocsparse_int i = 0; i < n; ++i)
        {
            for(rocsparse_int j = 0; j < n; ++j)
            {
                const double val = (j < i) ? LU[i * n + j] : (j == i);
                std::cout << std::setw(3) << val;

                errors += std::fabs(val - L_expected[i * n + j]) > tol;
            }
            std::cout << "\n";
        }
        std::cout << std::endl;

        std::cout << "and U matrix: \n";

        for(rocsparse_int i = 0; i < n; ++i)
        {
            for(rocsparse_int j = 0; j < n; ++j)
            {
                const double val = (j >= i) ? LU[i * n + j] : 0;
                std::cout << std::setw(3) << val;

                errors += std::fabs(val - U_expected[i * n + j]) > tol;
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    // 10. Free rocSPARSE resources and device memory.
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_LU));
    HIP_CHECK(hipFree(perm));
    HIP_CHECK(hipFree(sort_temp_buffer));
    HIP_CHECK(hipFree(temp_buffer));

    // 11. Print validation result.
    return report_validation_result(errors);
}
