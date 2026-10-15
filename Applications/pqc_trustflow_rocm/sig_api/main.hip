// MIT License
//
// Copyright (c) 2026 firedoil
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

// HIP example for batched ML-DSA and Aigis-sig signatures.

#include <hip/hip_runtime.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "batch_keygen.hpp"
#include "batch_sign.hpp"
#include "batch_verify.hpp"
#include "config.h"
#include "params.h"
#include "sign.hpp"

namespace
{

constexpr unsigned kSignatureNonceValues = static_cast<unsigned>(UINT16_MAX) + 1u;
static_assert(kSignatureNonceValues / PARAM_L >= MAX_SIGN_ROUNDS,
              "The signature nonce space is too small for the configured signing rounds");
constexpr int kMaxBatchCount
    = static_cast<int>(kSignatureNonceValues / PARAM_L - MAX_SIGN_ROUNDS + 1u);

int check_hip(hipError_t error, const char* expression, const char* file, int line)
{
    if(error == hipSuccess)
        return 0;

    std::fprintf(stderr,
                 "HIP error at %s:%d while evaluating %s: %s\n",
                 file,
                 line,
                 expression,
                 hipGetErrorString(error));
    return 1;
}

#define HIP_CHECK(call)                                       \
    do                                                        \
    {                                                         \
        if(check_hip((call), #call, __FILE__, __LINE__) != 0) \
            return 1;                                         \
    }                                                         \
    while(0)

template<typename T>
class DeviceBuffer
{
public:
    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if(data_ != nullptr)
            hipFree(data_);
    }

    hipError_t allocate(std::size_t count)
    {
        return hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    }

    T* get()
    {
        return data_;
    }
    const T* get() const
    {
        return data_;
    }

private:
    T* data_ = nullptr;
};

class BatchKeygenHandle
{
public:
    ~BatchKeygenHandle()
    {
        batch_keygen_free(&buffers);
    }
    int allocate(int batch_count)
    {
        return batch_keygen_alloc(&buffers, batch_count);
    }

    BatchKeygenBuffers buffers{};
};

class BatchSignHandle
{
public:
    ~BatchSignHandle()
    {
        batch_sign_free(&pipeline);
    }
    int allocate(int batch_count)
    {
        return batch_sign_alloc(&pipeline, batch_count);
    }

    BatchSignPipeline pipeline{};
};

class BatchVerifyHandle
{
public:
    ~BatchVerifyHandle()
    {
        batch_verify_free(&buffers);
    }
    int allocate(int batch_count)
    {
        return batch_verify_alloc(&buffers, batch_count);
    }

    BatchVerifyBuffers buffers{};
};

enum class Mode
{
    self_test,
    sign,
    verify,
};

void print_usage(const char* program)
{
    std::printf("Usage:\n"
                "  %s [--self-test]\n"
                "  %s --api-sig-sign --msg-in FILE --pk-out FILE --sk-out FILE "
                "--sig-out FILE [--batch N]\n"
                "  %s --api-sig-verify --msg-in FILE --pk-in FILE --sig-in FILE "
                "[--batch N]\n",
                program,
                program,
                program);
}

const char* argument_value(int argc, char** argv, const char* name)
{
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::strcmp(argv[i], name) == 0)
            return argv[i + 1];
    }
    return nullptr;
}

bool option_takes_value(const char* option)
{
    return std::strcmp(option, "--batch") == 0 || std::strcmp(option, "--msg-in") == 0
           || std::strcmp(option, "--pk-in") == 0 || std::strcmp(option, "--pk-out") == 0
           || std::strcmp(option, "--sk-out") == 0 || std::strcmp(option, "--sig-in") == 0
           || std::strcmp(option, "--sig-out") == 0;
}

int parse_batch_count(const char* value, int* batch_count)
{
    char* end         = nullptr;
    errno             = 0;
    const long parsed = std::strtol(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > kMaxBatchCount)
    {
        std::fprintf(stderr,
                     "Invalid batch size: %s (expected 1 through %d for this parameter set).\n",
                     value,
                     kMaxBatchCount);
        return 2;
    }
    *batch_count = static_cast<int>(parsed);
    return 0;
}

int parse_mode(int argc, char** argv, Mode* mode, int* batch_count)
{
    bool mode_selected = false;
    *mode              = Mode::self_test;
    *batch_count       = 1;

    for(int i = 1; i < argc; ++i)
    {
        const char* argument      = argv[i];
        Mode        selected_mode = Mode::self_test;
        bool        is_mode       = true;

        if(std::strcmp(argument, "--self-test") == 0)
            selected_mode = Mode::self_test;
        else if(std::strcmp(argument, "--api-sig-sign") == 0)
            selected_mode = Mode::sign;
        else if(std::strcmp(argument, "--api-sig-verify") == 0)
            selected_mode = Mode::verify;
        else
            is_mode = false;

        if(is_mode)
        {
            if(mode_selected)
            {
                std::fprintf(stderr, "Select exactly one operation mode.\n");
                return 2;
            }
            *mode         = selected_mode;
            mode_selected = true;
            continue;
        }

        if(std::strcmp(argument, "-h") == 0 || std::strcmp(argument, "--help") == 0)
        {
            print_usage(argv[0]);
            return 1;
        }

        if(option_takes_value(argument))
        {
            if(i + 1 >= argc)
            {
                std::fprintf(stderr, "Missing value for %s\n", argument);
                return 2;
            }
            if(std::strcmp(argument, "--batch") == 0)
            {
                const int result = parse_batch_count(argv[i + 1], batch_count);
                if(result != 0)
                    return result;
            }
            ++i;
            continue;
        }

        std::fprintf(stderr, "Unknown option: %s\n", argument);
        return 2;
    }

    return 0;
}

int read_file(const char* path, std::vector<uint8_t>* output)
{
    std::FILE* file = std::fopen(path, "rb");
    if(file == nullptr)
    {
        std::fprintf(stderr, "Could not open %s for reading.\n", path);
        return 2;
    }

    if(std::fseek(file, 0, SEEK_END) != 0)
    {
        std::fclose(file);
        return 2;
    }
    const long size = std::ftell(file);
    if(size < 0 || std::fseek(file, 0, SEEK_SET) != 0)
    {
        std::fclose(file);
        return 2;
    }

    output->resize(static_cast<std::size_t>(size));
    const std::size_t bytes_read
        = output->empty() ? 0 : std::fread(output->data(), 1, output->size(), file);
    std::fclose(file);
    if(bytes_read != output->size())
    {
        std::fprintf(stderr, "Could not read %s.\n", path);
        return 2;
    }
    return 0;
}

int read_exact_file(const char* path, std::vector<uint8_t>* output, std::size_t size)
{
    const int result = read_file(path, output);
    if(result != 0)
        return result;
    if(output->size() != size)
    {
        std::fprintf(stderr,
                     "Unexpected file size for %s: expected exactly %zu bytes.\n",
                     path,
                     size);
        return 2;
    }
    return 0;
}

int write_file(const char* path, const uint8_t* data, std::size_t size)
{
    std::FILE* file = std::fopen(path, "wb");
    if(file == nullptr)
    {
        std::fprintf(stderr, "Could not open %s for writing.\n", path);
        return 2;
    }

    const std::size_t bytes_written = std::fwrite(data, 1, size, file);
    const int         close_result  = std::fclose(file);
    if(bytes_written != size || close_result != 0)
    {
        std::fprintf(stderr, "Could not write %s.\n", path);
        return 2;
    }
    return 0;
}

int fill_random(std::vector<uint8_t>* output)
{
    std::FILE* file = std::fopen("/dev/urandom", "rb");
    if(file == nullptr)
    {
        std::fprintf(stderr, "Could not open the system random source.\n");
        return 2;
    }

    const std::size_t bytes_read = std::fread(output->data(), 1, output->size(), file);
    std::fclose(file);
    if(bytes_read != output->size())
    {
        std::fprintf(stderr, "Could not read enough random data.\n");
        return 2;
    }
    return 0;
}

void repeat_record(std::vector<uint8_t>*       output,
                   const std::vector<uint8_t>& record,
                   int                         batch_count)
{
    output->resize(record.size() * static_cast<std::size_t>(batch_count));
    for(int i = 0; i < batch_count; ++i)
    {
        std::memcpy(output->data() + static_cast<std::size_t>(i) * record.size(),
                    record.data(),
                    record.size());
    }
}

void build_pre(std::vector<uint8_t>* pre)
{
#if ALGORITHM == ALGO_MLDSA
    pre->assign(2, 0);
#else
    pre->assign(1, 0);
#endif
}

std::size_t pre_size(const std::vector<uint8_t>& pre)
{
#if ALGORITHM == ALGO_MLDSA
    return pre.size();
#else
    (void)pre;
    return 0;
#endif
}

int run_batch_signature_test(const uint8_t*              seed,
                             const uint8_t*              random,
                             const std::vector<uint8_t>& message,
                             const std::vector<uint8_t>& pre)
{
    constexpr int batch_count = 2;

    BatchKeygenHandle keygen;
    BatchSignHandle   signer;
    BatchVerifyHandle verifier;
    if(keygen.allocate(batch_count) != 0 || signer.allocate(batch_count) != 0
       || verifier.allocate(batch_count) != 0)
    {
        std::fprintf(stderr, "Could not allocate signature batch self-test buffers.\n");
        return 1;
    }

    DeviceBuffer<uint8_t>   device_seed;
    DeviceBuffer<precomp_t> precomputation;
    DeviceBuffer<uint8_t>   device_message;
    DeviceBuffer<uint8_t>   device_messages;
    DeviceBuffer<uint8_t>   device_pre;
    DeviceBuffer<uint8_t>   device_random;
    HIP_CHECK(device_seed.allocate(SEEDBYTES));
    HIP_CHECK(precomputation.allocate(1));
    HIP_CHECK(device_message.allocate(message.empty() ? 1 : message.size()));
    HIP_CHECK(device_pre.allocate(pre.size()));
    HIP_CHECK(device_random.allocate(RNDBYTES > 0 ? RNDBYTES : 1));
    HIP_CHECK(hipMemcpy(device_seed.get(), seed, SEEDBYTES, hipMemcpyHostToDevice));
    if(!message.empty())
    {
        HIP_CHECK(
            hipMemcpy(device_message.get(), message.data(), message.size(), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(device_pre.get(), pre.data(), pre.size(), hipMemcpyHostToDevice));
#if RNDBYTES > 0
    HIP_CHECK(hipMemcpy(device_random.get(), random, RNDBYTES, hipMemcpyHostToDevice));
#else
    (void)random;
#endif

    if(batch_keygen_pipeline_warp_opt(keygen.buffers.d_pks,
                                      keygen.buffers.d_sks,
                                      device_seed.get(),
                                      &keygen.buffers,
                                      batch_count,
                                      nullptr,
                                      0,
                                      1)
       != 0)
    {
        std::fprintf(stderr, "Signature batch key generation self-test failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());

    batch_keygen_material_to_precomp_kernel<<<1, 1>>>(precomputation.get(),
                                                       keygen.buffers.d_mat,
                                                       keygen.buffers.d_s1hat,
                                                       keygen.buffers.d_s2_ntt,
                                                       keygen.buffers.d_t0_ntt,
                                                       keygen.buffers.d_buf,
                                                       keygen.buffers.d_tr,
                                                       0,
                                                       0);
    HIP_CHECK(hipGetLastError());

    int completed = 0;
    if(batch_sign_pipeline_ex(&signer.pipeline,
                              batch_count,
                              precomputation.get(),
                              device_message.get(),
                              message.size(),
                              device_pre.get(),
                              pre_size(pre),
                              device_random.get(),
                              nullptr,
                              nullptr,
                              &completed)
           != 0
       || completed != batch_count)
    {
        std::fprintf(stderr, "Signature batch signing self-test failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint8_t> signatures(static_cast<std::size_t>(batch_count) * CRYPTO_BYTES);
    HIP_CHECK(hipMemcpy(signatures.data(),
                        signer.pipeline.d_sigs,
                        signatures.size(),
                        hipMemcpyDeviceToHost));

    std::vector<uint8_t> messages;
    repeat_record(&messages, message, batch_count);
    HIP_CHECK(device_messages.allocate(messages.empty() ? 1 : messages.size()));
    if(!messages.empty())
    {
        HIP_CHECK(hipMemcpy(device_messages.get(),
                            messages.data(),
                            messages.size(),
                            hipMemcpyHostToDevice));
    }

    launch_batch_verify_precompute(verifier.buffers.d_mat,
                                   verifier.buffers.d_t1_hat,
                                   verifier.buffers.d_tr,
                                   keygen.buffers.d_pks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> results(batch_count, -1);
    if(batch_verify_pipeline(&verifier.buffers,
                             signatures.data(),
                             device_messages.get(),
                             message.size(),
                             device_pre.get(),
                             pre_size(pre),
                             batch_count,
                             results.data())
       != 0)
    {
        std::fprintf(stderr, "Signature batch verification self-test failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    for(int result : results)
    {
        if(result != 0)
        {
            std::fprintf(stderr, "Signature batch verification self-test rejected valid data.\n");
            return 1;
        }
    }

    signatures[0] ^= 1;
    if(batch_verify_pipeline(&verifier.buffers,
                             signatures.data(),
                             device_messages.get(),
                             message.size(),
                             device_pre.get(),
                             pre_size(pre),
                             batch_count,
                             results.data())
       != 0)
    {
        std::fprintf(stderr, "Signature batch tamper self-test failed to run.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    if(results[0] == 0)
    {
        std::fprintf(stderr, "Signature batch tamper self-test accepted modified data.\n");
        return 1;
    }
    for(int i = 1; i < batch_count; ++i)
    {
        if(results[i] != 0)
        {
            std::fprintf(stderr, "Signature batch tamper self-test rejected unchanged data.\n");
            return 1;
        }
    }

    return 0;
}

int run_self_test()
{
    constexpr std::size_t message_size = 32;
    constexpr std::size_t random_size  = RNDBYTES > 0 ? RNDBYTES : 1;

    std::array<uint8_t, SEEDBYTES>   seed{};
    std::array<uint8_t, random_size> random{};
    std::vector<uint8_t>             message(message_size);
    std::vector<uint8_t>             pre;
    build_pre(&pre);

    for(std::size_t i = 0; i < seed.size(); ++i)
        seed[i] = static_cast<uint8_t>((13u * i + 5u) & 0xffu);
    for(std::size_t i = 0; i < random.size(); ++i)
        random[i] = static_cast<uint8_t>((19u * i + 11u) & 0xffu);
    for(std::size_t i = 0; i < message.size(); ++i)
        message[i] = static_cast<uint8_t>((23u * i + 3u) & 0xffu);

    if(run_batch_signature_test(seed.data(), random.data(), message, pre) != 0)
        return 1;

    std::printf("Signature self-test PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int run_sign(int argc, char** argv, int batch_count)
{
    const char* message_path    = argument_value(argc, argv, "--msg-in");
    const char* public_key_path = argument_value(argc, argv, "--pk-out");
    const char* secret_key_path = argument_value(argc, argv, "--sk-out");
    const char* signature_path  = argument_value(argc, argv, "--sig-out");
    if(message_path == nullptr || public_key_path == nullptr || secret_key_path == nullptr
       || signature_path == nullptr)
    {
        std::fprintf(stderr,
                     "Signing requires --msg-in, --pk-out, --sk-out, and "
                     "--sig-out.\n");
        return 2;
    }

    std::vector<uint8_t> message;
    if(read_file(message_path, &message) != 0)
        return 2;
    std::vector<uint8_t> seed(SEEDBYTES);
    std::vector<uint8_t> random(RNDBYTES > 0 ? RNDBYTES : 1);
    std::vector<uint8_t> pre;
    build_pre(&pre);
    if(fill_random(&seed) != 0 || fill_random(&random) != 0)
        return 2;

    BatchKeygenHandle keygen;
    BatchSignHandle   signer;
    if(keygen.allocate(batch_count) != 0 || signer.allocate(batch_count) != 0)
    {
        std::fprintf(stderr, "Could not allocate signature batch buffers.\n");
        return 1;
    }

    DeviceBuffer<uint8_t>   device_seed;
    DeviceBuffer<uint8_t>   public_key;
    DeviceBuffer<uint8_t>   secret_key;
    DeviceBuffer<precomp_t> precomputation;
    DeviceBuffer<uint8_t>   device_message;
    DeviceBuffer<uint8_t>   device_pre;
    DeviceBuffer<uint8_t>   device_random;
    HIP_CHECK(device_seed.allocate(seed.size()));
    HIP_CHECK(public_key.allocate(CRYPTO_PUBLICKEYBYTES));
    HIP_CHECK(secret_key.allocate(CRYPTO_SECRETKEYBYTES));
    HIP_CHECK(precomputation.allocate(1));
    HIP_CHECK(device_message.allocate(message.empty() ? 1 : message.size()));
    HIP_CHECK(device_pre.allocate(pre.size()));
    HIP_CHECK(device_random.allocate(random.size()));
    HIP_CHECK(hipMemcpy(device_seed.get(), seed.data(), seed.size(), hipMemcpyHostToDevice));

    if(batch_keygen_pipeline_warp_opt(keygen.buffers.d_pks,
                                      keygen.buffers.d_sks,
                                      device_seed.get(),
                                      &keygen.buffers,
                                      batch_count,
                                      nullptr,
                                      0,
                                      1)
       != 0)
    {
        std::fprintf(stderr, "Batch key generation failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(public_key.get(),
                        keygen.buffers.d_pks,
                        CRYPTO_PUBLICKEYBYTES,
                        hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(secret_key.get(),
                        keygen.buffers.d_sks,
                        CRYPTO_SECRETKEYBYTES,
                        hipMemcpyDeviceToDevice));

    batch_keygen_material_to_precomp_kernel<<<1, 1>>>(precomputation.get(),
                                                       keygen.buffers.d_mat,
                                                       keygen.buffers.d_s1hat,
                                                       keygen.buffers.d_s2_ntt,
                                                       keygen.buffers.d_t0_ntt,
                                                       keygen.buffers.d_buf,
                                                       keygen.buffers.d_tr,
                                                       0,
                                                       0);
    HIP_CHECK(hipGetLastError());
    if(!message.empty())
    {
        HIP_CHECK(
            hipMemcpy(device_message.get(), message.data(), message.size(), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(device_pre.get(), pre.data(), pre.size(), hipMemcpyHostToDevice));
#if RNDBYTES > 0
    HIP_CHECK(hipMemcpy(device_random.get(), random.data(), random.size(), hipMemcpyHostToDevice));
#endif

    int completed = 0;
    if(batch_sign_pipeline_ex(&signer.pipeline,
                              batch_count,
                              precomputation.get(),
                              device_message.get(),
                              message.size(),
                              device_pre.get(),
                              pre_size(pre),
                              device_random.get(),
                              nullptr,
                              nullptr,
                              &completed)
           != 0
       || completed != batch_count)
    {
        std::fprintf(stderr, "Batch signing failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint8_t> host_public_key(CRYPTO_PUBLICKEYBYTES);
    std::vector<uint8_t> host_secret_key(CRYPTO_SECRETKEYBYTES);
    std::vector<uint8_t> host_signature(CRYPTO_BYTES);
    HIP_CHECK(hipMemcpy(host_public_key.data(),
                        public_key.get(),
                        host_public_key.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_secret_key.data(),
                        secret_key.get(),
                        host_secret_key.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_signature.data(),
                        signer.pipeline.d_sigs,
                        host_signature.size(),
                        hipMemcpyDeviceToHost));

    if(write_file(public_key_path, host_public_key.data(), host_public_key.size()) != 0
       || write_file(secret_key_path, host_secret_key.data(), host_secret_key.size()) != 0
       || write_file(signature_path, host_signature.data(), host_signature.size()) != 0)
        return 2;

    std::printf("Signature generation PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int run_verify(int argc, char** argv, int batch_count)
{
    const char* message_path    = argument_value(argc, argv, "--msg-in");
    const char* public_key_path = argument_value(argc, argv, "--pk-in");
    const char* signature_path  = argument_value(argc, argv, "--sig-in");
    if(message_path == nullptr || public_key_path == nullptr || signature_path == nullptr)
    {
        std::fprintf(stderr, "Verification requires --msg-in, --pk-in, and --sig-in.\n");
        return 2;
    }

    std::vector<uint8_t> message;
    std::vector<uint8_t> public_key;
    std::vector<uint8_t> signature;
    if(read_file(message_path, &message) != 0
       || read_exact_file(public_key_path, &public_key, CRYPTO_PUBLICKEYBYTES) != 0
       || read_exact_file(signature_path, &signature, CRYPTO_BYTES) != 0)
        return 2;

    std::vector<uint8_t> signatures;
    std::vector<uint8_t> messages;
    repeat_record(&signatures, signature, batch_count);
    repeat_record(&messages, message, batch_count);
    std::vector<uint8_t> pre;
    build_pre(&pre);
    std::vector<int> results(static_cast<std::size_t>(batch_count), -1);

    BatchVerifyHandle verifier;
    if(verifier.allocate(batch_count) != 0)
    {
        std::fprintf(stderr, "Could not allocate verification batch buffers.\n");
        return 1;
    }

    DeviceBuffer<uint8_t> device_public_key;
    DeviceBuffer<uint8_t> device_messages;
    DeviceBuffer<uint8_t> device_pre;
    HIP_CHECK(device_public_key.allocate(public_key.size()));
    HIP_CHECK(device_messages.allocate(messages.empty() ? 1 : messages.size()));
    HIP_CHECK(device_pre.allocate(pre.size()));
    HIP_CHECK(hipMemcpy(device_public_key.get(),
                        public_key.data(),
                        public_key.size(),
                        hipMemcpyHostToDevice));
    if(!messages.empty())
    {
        HIP_CHECK(hipMemcpy(device_messages.get(),
                            messages.data(),
                            messages.size(),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(device_pre.get(), pre.data(), pre.size(), hipMemcpyHostToDevice));

    launch_batch_verify_precompute(verifier.buffers.d_mat,
                                   verifier.buffers.d_t1_hat,
                                   verifier.buffers.d_tr,
                                   device_public_key.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(batch_verify_pipeline(&verifier.buffers,
                             signatures.data(),
                             device_messages.get(),
                             message.size(),
                             device_pre.get(),
                             pre_size(pre),
                             batch_count,
                             results.data())
       != 0)
    {
        std::fprintf(stderr, "Batch verification failed.\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(int result : results)
    {
        if(result != 0)
        {
            std::fprintf(stderr, "Signature verification failed.\n");
            return 1;
        }
    }

    std::printf("Signature verification PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int print_device()
{
    int             device = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::printf("GPU: %s (%s, %d compute units)\n",
                properties.name,
                properties.gcnArchName,
                properties.multiProcessorCount);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Mode      mode         = Mode::self_test;
    int       batch_count  = 1;
    const int parse_result = parse_mode(argc, argv, &mode, &batch_count);
    if(parse_result == 1)
        return 0;
    if(parse_result != 0)
    {
        print_usage(argv[0]);
        return parse_result;
    }

    if(print_device() != 0)
        return 1;

    const hipError_t stack_result = hipDeviceSetLimit(hipLimitStackSize, 128u * 1024u);
    if(stack_result != hipSuccess)
    {
        std::fprintf(stderr,
                     "Warning: could not set the HIP device stack size: %s\n",
                     hipGetErrorString(stack_result));
        hipGetLastError();
    }

    switch(mode)
    {
        case Mode::self_test: return run_self_test();
        case Mode::sign: return run_sign(argc, argv, batch_count);
        case Mode::verify: return run_verify(argc, argv, batch_count);
    }

    return 2;
}
