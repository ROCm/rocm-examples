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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "CmdParser/cmdparser.hpp"
#include "rocprofiler_utils.hpp"

#include <libgen.h>
#include <thread>

namespace
{
using auto_lock_t                      = std::unique_lock<std::mutex>;
auto               print_lock          = std::mutex{};
size_t             nthread_per_device  = 2;
size_t             nitr                = 500;
size_t             nsync               = 10;
constexpr unsigned shared_mem_tile_dim = 32;

void verify(int* in, int* out, int M, int N);
} // namespace

__global__ void transpose(const int* in, int* out, int M, int N);

void run(int rank, int tid, int devid, int argc, char** argv);

void run_transpose(int rank, int tid, hipStream_t stream, int argc, char** argv);

void run_migrate(int rank, int tid, hipStream_t stream, int, char** argv);

void run_scratch(int rank, int tid, hipStream_t stream, int argc, char** argv);

int main(int argc, char** argv)
{
    auto* exe_name = basename(argv[0]);

    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("n", "nthreads", 2, "Number of threads per device");
    parser.set_optional<size_t>("i", "iterations", 500, "Number of iterations");
    parser.set_optional<size_t>("s", "sync", 10, "Sync every N iterations");
    parser.run_and_exit_if_error();

    nthread_per_device = parser.get<size_t>("n");
    nitr               = parser.get<size_t>("i");
    nsync              = parser.get<size_t>("s");

    int ndevice = 0;
    HIP_CHECK(hipGetDeviceCount(&ndevice));

    auto nthreads = (ndevice * nthread_per_device);

    common::safe_printer::printf("[%s] Number of devices found: %i\n", exe_name, ndevice);
    common::safe_printer::printf("[%s] Number of threads (per device): %zu\n",
                                 exe_name,
                                 nthread_per_device);
    common::safe_printer::printf("[%s] Number of threads (total): %zu\n", exe_name, nthreads);
    common::safe_printer::printf("[%s] Number of iterations: %zu\n", exe_name, nitr);
    common::safe_printer::printf("[%s] Syncing every %zu iterations\n", exe_name, nsync);

    {
        auto _threads = std::vector<std::thread>{};
        for(size_t i = 0; i < nthreads; ++i)
        {
            _threads.emplace_back(run, 0, i, i % ndevice, argc, argv);
        }
        for(auto& itr : _threads)
        {
            itr.join();
        }
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipDeviceReset());

    return 0;
}

__global__ void transpose(const int* in, int* out, int M, int N)
{
    __shared__ int tile[shared_mem_tile_dim][shared_mem_tile_dim];

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * M + blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    idx      = (blockIdx.x * blockDim.x + threadIdx.y) * N + blockIdx.y * blockDim.y + threadIdx.x;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

template<typename Tp>
__global__ void test_page_migrate(Tp* data, Tp val)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    data[idx] += val;
}

__global__ void test_kern_large(uint64_t* output)
{
    uint64_t result = 0;
    int      test[4000];
    memset(test, 5, 4000);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void test_kern_medium(uint64_t* output)
{
    uint64_t result = 0;
    int      test[175];
    memset(test, 5, 175);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void test_kern_small(uint64_t* output)
{
    uint64_t result = 0;
    int      test[2];
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

void run(int rank, int tid, int devid, int argc, char** argv)
{
    auto* stream = hipStream_t{};
    HIP_CHECK(hipSetDevice(devid));
    HIP_CHECK(hipStreamCreate(&stream));

    run_migrate(rank, tid, stream, argc, argv);
    run_scratch(rank, tid, stream, argc, argv);
    run_transpose(rank, tid, stream, argc, argv);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
}

void run_transpose(int rank, int tid, hipStream_t stream, int, char** argv)
{
    auto* exe_name = basename(argv[0]);

    unsigned int M = 4960 * 2;
    unsigned int N = 4960 * 2;

    auto_lock_t _lk{print_lock};
    common::safe_printer::printf("[%s][transpose][%d][%d] M: %u N: %u\n",
                                 exe_name,
                                 rank,
                                 tid,
                                 M,
                                 N);
    _lk.unlock();

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
    dim3 block(32, 32, 1); // transpose

    common::safe_printer::printf("[%s][transpose][%i][%i] grid=(%i,%i,%i), block=(%i,%i,%i)\n",
                                 exe_name,
                                 rank,
                                 tid,
                                 grid.x,
                                 grid.y,
                                 grid.z,
                                 block.x,
                                 block.y,
                                 block.z);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitr; ++i)
    {
        transpose<<<grid, block, 0, stream>>>(in, out, M, N);
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

    common::safe_printer::printf("[%s][transpose][%d][%d] Runtime of transpose is %f sec\n",
                                 exe_name,
                                 rank,
                                 tid,
                                 time);
    common::safe_printer::printf(
        "[%s][transpose][%d][%d] The average performance of transpose is %f GBytes/sec\n",
        exe_name,
        rank,
        tid,
        GB / time);

    HIP_CHECK(hipStreamSynchronize(stream));

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(inp_matrix, out_matrix, M, N);

    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    delete[] inp_matrix;
    delete[] out_matrix;
}

void run_scratch(int rank, int tid, hipStream_t stream, int, char** argv)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_CHECK(hipStreamSynchronize(stream));

    const auto* exe_name = basename(argv[0]);

    uint64_t* data_ptr = nullptr;
    HIP_CHECK(HIP_HOST_ALLOC_FUNC(&data_ptr, sizeof(uint64_t), 0));
    *data_ptr = 0;

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_medium<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_large<<<1100, 1, 0, stream>>>(data_ptr);
    HIP_CHECK(hipStreamSynchronize(stream));

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_CHECK(hipStreamSynchronize(stream));

    test_kern_medium<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_CHECK(hipStreamSynchronize(stream));

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_CHECK(hipStreamSynchronize(stream));

    test_kern_large<<<1100, 1, 0, stream>>>(data_ptr);
    HIP_CHECK(hipStreamSynchronize(stream));

    auto   t2   = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    common::safe_printer::printf("[%s][scratch][%d][%d] Runtime of scratch is %f sec\n",
                                 exe_name,
                                 rank,
                                 tid,
                                 time);
}

void run_migrate(int rank, int tid, hipStream_t stream, int, char** argv)
{
    using data_type            = uint64_t;
    constexpr data_type init_v = 1;
    constexpr data_type incr_v = 1;

    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_CHECK(hipStreamSynchronize(stream));

    const auto* exe_name  = basename(argv[0]);
    auto        page_data = std::vector<data_type>(1024, 0);

    HIP_CHECK(hipHostRegister(page_data.data(),
                              page_data.size() * sizeof(data_type),
                              hipHostRegisterDefault));

    for(auto& itr : page_data)
    {
        itr = init_v;
    }

    auto page_data_dev_ptr = static_cast<data_type*>(nullptr);
    HIP_CHECK(hipHostGetDevicePointer(&page_data_dev_ptr, page_data.data(), 0));

    test_page_migrate<<<1, 1024, 0, stream>>>(page_data_dev_ptr, incr_v);

    HIP_CHECK(hipStreamSynchronize(stream));

    for(auto& itr : page_data)
    {
        auto diff = (itr - incr_v);
        if(diff != init_v)
        {
            auto msg = std::stringstream{};
            msg << "invalid diff: " << diff << ". expected: " << init_v;
            throw std::runtime_error{msg.str()};
        }
    }

    HIP_CHECK(hipHostUnregister(page_data.data()));

    auto   t2   = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    common::safe_printer::printf("[%s][migrate][%d][%d] Runtime of migrate is %f sec\n",
                                 exe_name,
                                 rank,
                                 tid,
                                 time);
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
