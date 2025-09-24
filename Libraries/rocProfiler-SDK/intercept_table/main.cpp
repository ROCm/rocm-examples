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

#include "CmdParser/cmdparser.hpp"
#include "client.hpp"
#include "rocprofiler_utils.hpp"

#include <thread>
#include <vector>

namespace
{
using auto_lock_t                      = std::unique_lock<std::mutex>;
auto               print_lock          = std::mutex{};
size_t             nthreads            = 2;
size_t             nitr                = 500;
size_t             nsync               = 10;
constexpr unsigned shared_mem_tile_dim = 32;

void verify(int* in, int* out, int M, int N);
} // namespace

__global__ void transpose_a(int* in, int* out, int M, int N);

void run(int rank, int tid, hipStream_t stream, int argc, char** argv);

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("n", "nthreads", 2, "Number of threads");
    parser.set_optional<size_t>("i", "iterations", 500, "Number of iterations");
    parser.set_optional<size_t>("s", "sync", 10, "Sync every N iterations");
    parser.run_and_exit_if_error();

    nthreads = parser.get<size_t>("n");
    nitr     = parser.get<size_t>("i");
    nsync    = parser.get<size_t>("s");

    common::safe_printer::printf("[transpose] Number of threads: %zu\n", nthreads);
    common::safe_printer::printf("[transpose] Number of iterations: %zu\n", nitr);
    common::safe_printer::printf("[transpose] Syncing every %zu iterations\n", nsync);

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
    int ndevice = 0;
    int devid   = 0;
    HIP_CHECK(hipGetDeviceCount(&ndevice));
    common::safe_printer::printf("[transpose] Number of devices found: %i\n", ndevice);
    if(ndevice > 0)
    {
        devid = 0 % ndevice;
        HIP_CHECK(hipSetDevice(devid));
        common::safe_printer::printf("[transpose] Rank %i assigned to device %i\n", 0, devid);
    }
    if(0 == devid && 0 < ndevice)
    {
        std::vector<std::thread> _threads{};
        std::vector<hipStream_t> _streams(nthreads);
        for(size_t i = 0; i < nthreads; ++i)
        {
            HIP_CHECK(hipStreamCreate(&_streams.at(i)));
        }
        for(size_t i = 1; i < nthreads; ++i)
        {
            _threads.emplace_back(run, 0, i, _streams.at(i), argc, argv);
        }
        run(0, 0, _streams.at(0), argc, argv);
        for(auto& itr : _threads)
        {
            itr.join();
        }
        for(size_t i = 0; i < nthreads; ++i)
        {
            HIP_CHECK(hipStreamDestroy(_streams.at(i)));
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipDeviceReset());

    return 0;
}

__global__ void transpose_a(int* in, int* out, int M, int N)
{
    __shared__ int tile[shared_mem_tile_dim][shared_mem_tile_dim];

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * M + blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    idx      = (blockIdx.x * blockDim.x + threadIdx.y) * N + blockIdx.y * blockDim.y + threadIdx.x;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

void run(int rank, int tid, hipStream_t stream, int, char**)
{
    unsigned int M = 4960 * 2;
    unsigned int N = 4960 * 2;

    common::safe_printer::printf("[transpose][%d][%d] M: %u N: %u\n", rank, tid, M, N);

    std::default_random_engine         _engine{std::random_device{}() * (rank + 1) * (tid + 1)};
    std::uniform_int_distribution<int> _dist{0, 1000};

    size_t size       = sizeof(int) * M * N;
    int*   inp_matrix = new int[size];
    int*   out_matrix = new int[size];
    for(size_t i = 0; i < M * N; i++)
    {
        inp_matrix[i] = _dist(_engine);
        out_matrix[i] = 0;
    }
    int* in  = nullptr;
    int* out = nullptr;

    HIP_CHECK(hipMalloc(&in, size));
    HIP_CHECK(hipMalloc(&out, size));
    HIP_CHECK(hipMemsetAsync(in, 0, size, stream));
    HIP_CHECK(hipMemsetAsync(out, 0, size, stream));
    HIP_CHECK(hipMemcpyAsync(in, inp_matrix, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1); // transpose_a

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitr; ++i)
    {
        transpose_a<<<grid, block, 0, stream>>>(in, out, M, N);
        HIP_CHECK(hipGetLastError());
        if(i % nsync == (nsync - 1))
        {
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float  GB   = (float)size * nitr * 2 / (1 << 30);

    common::safe_printer::printf("[%d][%d] Runtime of transpose is %f sec\n", rank, tid, time);
    common::safe_printer::printf("The average performance of transpose is %f GBytes/sec\n",
                                 GB / time);

    HIP_CHECK(hipStreamSynchronize(stream));

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(inp_matrix, out_matrix, M, N);

    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    delete[] inp_matrix;
    delete[] out_matrix;
}

namespace
{
void verify(int* in, int* out, int M, int N)
{
    for(int i = 0; i < 10; i++)
    {
        int row = rand() % M;
        int col = rand() % N;
        if(in[row * N + col] != out[col * M + row])
        {
            auto_lock_t _lk{print_lock};
            common::safe_printer::printf("mismatch: %d, %d : %d | %d\n",
                                         row,
                                         col,
                                         in[row * N + col],
                                         out[col * M + row]);
        }
    }
}
} // namespace
