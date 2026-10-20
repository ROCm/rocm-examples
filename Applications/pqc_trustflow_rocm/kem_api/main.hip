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

// HIP example for batched Kyber and Aigis-enc key encapsulation.

#include <hip/hip_runtime.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "batch_kem.hpp"
#include "config.h"
#include "params.h"

namespace
{

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

enum class Mode
{
    self_test,
    keygen,
    encaps,
    decaps,
};

void print_usage(const char* program)
{
    std::printf("Usage:\n"
                "  %s [--self-test]\n"
                "  %s --api-kem-keygen --pk-out FILE --sk-out FILE [--batch N]\n"
                "  %s --api-kem-encaps --pk-in FILE --ct-out FILE --ss-out FILE [--batch N]\n"
                "  %s --api-kem-decaps --sk-in FILE --ct-in FILE --ss-out FILE [--batch N]\n",
                program,
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
    return std::strcmp(option, "--batch") == 0 || std::strcmp(option, "--pk-in") == 0
           || std::strcmp(option, "--pk-out") == 0 || std::strcmp(option, "--sk-in") == 0
           || std::strcmp(option, "--sk-out") == 0 || std::strcmp(option, "--ct-in") == 0
           || std::strcmp(option, "--ct-out") == 0 || std::strcmp(option, "--ss-out") == 0;
}

int parse_batch_count(const char* value, int* batch_count)
{
    char* end         = nullptr;
    errno             = 0;
    const long parsed = std::strtol(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > INT_MAX)
    {
        std::fprintf(stderr, "Invalid batch size: %s\n", value);
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
        else if(std::strcmp(argument, "--api-kem-keygen") == 0)
            selected_mode = Mode::keygen;
        else if(std::strcmp(argument, "--api-kem-encaps") == 0)
            selected_mode = Mode::encaps;
        else if(std::strcmp(argument, "--api-kem-decaps") == 0)
            selected_mode = Mode::decaps;
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

int read_file(const char* path, std::vector<uint8_t>* output, std::size_t expected_size)
{
    std::FILE* file = std::fopen(path, "rb");
    if(file == nullptr)
    {
        std::fprintf(stderr, "Could not open %s for reading.\n", path);
        return 2;
    }

    output->resize(expected_size);
    const std::size_t bytes_read    = std::fread(output->data(), 1, expected_size, file);
    const int         trailing_byte = std::fgetc(file);
    std::fclose(file);

    if(bytes_read != expected_size || trailing_byte != EOF)
    {
        std::fprintf(stderr,
                     "Unexpected file size for %s: expected exactly %zu bytes.\n",
                     path,
                     expected_size);
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

int run_self_test()
{
    std::array<uint8_t, 2 * PARAM_SYMBYTES> keygen_coins{};
    std::array<uint8_t, PARAM_SYMBYTES>     encaps_coins{};
    std::array<uint8_t, PARAM_SSBYTES>      encapsulated_secret{};
    std::array<uint8_t, PARAM_SSBYTES>      decapsulated_secret{};

    for(std::size_t i = 0; i < keygen_coins.size(); ++i)
        keygen_coins[i] = static_cast<uint8_t>((17u * i + 29u) & 0xffu);
    for(std::size_t i = 0; i < encaps_coins.size(); ++i)
        encaps_coins[i] = static_cast<uint8_t>((31u * i + 7u) & 0xffu);

    DeviceBuffer<uint8_t> public_key;
    DeviceBuffer<uint8_t> secret_key;
    DeviceBuffer<uint8_t> ciphertext;
    DeviceBuffer<uint8_t> secret_from_encaps;
    DeviceBuffer<uint8_t> secret_from_decaps;
    DeviceBuffer<uint8_t> device_keygen_coins;
    DeviceBuffer<uint8_t> device_encaps_coins;

    HIP_CHECK(public_key.allocate(PARAM_PUBLICKEYBYTES));
    HIP_CHECK(secret_key.allocate(PARAM_SECRETKEYBYTES));
    HIP_CHECK(ciphertext.allocate(PARAM_CIPHERTEXTBYTES));
    HIP_CHECK(secret_from_encaps.allocate(PARAM_SSBYTES));
    HIP_CHECK(secret_from_decaps.allocate(PARAM_SSBYTES));
    HIP_CHECK(device_keygen_coins.allocate(keygen_coins.size()));
    HIP_CHECK(device_encaps_coins.allocate(encaps_coins.size()));

    HIP_CHECK(hipMemcpy(device_keygen_coins.get(),
                        keygen_coins.data(),
                        keygen_coins.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_encaps_coins.get(),
                        encaps_coins.data(),
                        encaps_coins.size(),
                        hipMemcpyHostToDevice));

    batch_kem_keypair_serial_kernel<<<1, 1>>>(public_key.get(),
                                              secret_key.get(),
                                              device_keygen_coins.get(),
                                              1);
    HIP_CHECK(hipGetLastError());
    batch_kem_encaps_serial_kernel<<<1, 1>>>(ciphertext.get(),
                                             secret_from_encaps.get(),
                                             public_key.get(),
                                             device_encaps_coins.get(),
                                             1);
    HIP_CHECK(hipGetLastError());
    batch_kem_decaps_serial_kernel<<<1, 1>>>(secret_from_decaps.get(),
                                             ciphertext.get(),
                                             secret_key.get(),
                                             1);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(encapsulated_secret.data(),
                        secret_from_encaps.get(),
                        encapsulated_secret.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(decapsulated_secret.data(),
                        secret_from_decaps.get(),
                        decapsulated_secret.size(),
                        hipMemcpyDeviceToHost));

    if(encapsulated_secret != decapsulated_secret)
    {
        std::fprintf(stderr, "KEM self-test failed: shared secrets differ.\n");
        return 1;
    }

    std::printf("KEM self-test PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int run_keygen(int argc, char** argv, int batch_count)
{
    const char* public_key_path = argument_value(argc, argv, "--pk-out");
    const char* secret_key_path = argument_value(argc, argv, "--sk-out");
    if(public_key_path == nullptr || secret_key_path == nullptr)
    {
        std::fprintf(stderr, "Key generation requires --pk-out and --sk-out.\n");
        return 2;
    }

    const std::size_t public_key_bytes
        = static_cast<std::size_t>(batch_count) * PARAM_PUBLICKEYBYTES;
    const std::size_t secret_key_bytes
        = static_cast<std::size_t>(batch_count) * PARAM_SECRETKEYBYTES;
    const std::size_t coin_bytes = static_cast<std::size_t>(batch_count) * 2 * PARAM_SYMBYTES;

    std::vector<uint8_t> host_public_key(PARAM_PUBLICKEYBYTES);
    std::vector<uint8_t> host_secret_key(PARAM_SECRETKEYBYTES);
    std::vector<uint8_t> host_coins(coin_bytes);
    if(fill_random(&host_coins) != 0)
        return 2;

    DeviceBuffer<uint8_t> public_keys;
    DeviceBuffer<uint8_t> secret_keys;
    DeviceBuffer<uint8_t> device_coins;
    HIP_CHECK(public_keys.allocate(public_key_bytes));
    HIP_CHECK(secret_keys.allocate(secret_key_bytes));
    HIP_CHECK(device_coins.allocate(coin_bytes));
    HIP_CHECK(hipMemcpy(device_coins.get(), host_coins.data(), coin_bytes, hipMemcpyHostToDevice));

    const int blocks = (batch_count + KEM_KEYGEN_TPB - 1) / KEM_KEYGEN_TPB;
    batch_kem_keypair_serial_kernel<<<blocks, KEM_KEYGEN_TPB>>>(public_keys.get(),
                                                                secret_keys.get(),
                                                                device_coins.get(),
                                                                batch_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host_public_key.data(),
                        public_keys.get(),
                        host_public_key.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_secret_key.data(),
                        secret_keys.get(),
                        host_secret_key.size(),
                        hipMemcpyDeviceToHost));

    if(write_file(public_key_path, host_public_key.data(), host_public_key.size()) != 0
       || write_file(secret_key_path, host_secret_key.data(), host_secret_key.size()) != 0)
        return 2;

    std::printf("KEM key generation PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int run_encaps(int argc, char** argv, int batch_count)
{
    const char* public_key_path = argument_value(argc, argv, "--pk-in");
    const char* ciphertext_path = argument_value(argc, argv, "--ct-out");
    const char* secret_path     = argument_value(argc, argv, "--ss-out");
    if(public_key_path == nullptr || ciphertext_path == nullptr || secret_path == nullptr)
    {
        std::fprintf(stderr, "Encapsulation requires --pk-in, --ct-out, and --ss-out.\n");
        return 2;
    }

    std::vector<uint8_t> public_key;
    if(read_file(public_key_path, &public_key, PARAM_PUBLICKEYBYTES) != 0)
        return 2;

    std::vector<uint8_t> public_keys;
    repeat_record(&public_keys, public_key, batch_count);
    std::vector<uint8_t> host_coins(static_cast<std::size_t>(batch_count) * PARAM_SYMBYTES);
    std::vector<uint8_t> ciphertext(PARAM_CIPHERTEXTBYTES);
    std::vector<uint8_t> shared_secret(PARAM_SSBYTES);
    if(fill_random(&host_coins) != 0)
        return 2;

    DeviceBuffer<uint8_t> device_public_keys;
    DeviceBuffer<uint8_t> device_ciphertexts;
    DeviceBuffer<uint8_t> device_secrets;
    DeviceBuffer<uint8_t> device_coins;
    HIP_CHECK(device_public_keys.allocate(public_keys.size()));
    HIP_CHECK(
        device_ciphertexts.allocate(static_cast<std::size_t>(batch_count) * PARAM_CIPHERTEXTBYTES));
    HIP_CHECK(device_secrets.allocate(static_cast<std::size_t>(batch_count) * PARAM_SSBYTES));
    HIP_CHECK(device_coins.allocate(host_coins.size()));
    HIP_CHECK(hipMemcpy(device_public_keys.get(),
                        public_keys.data(),
                        public_keys.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_coins.get(), host_coins.data(), host_coins.size(), hipMemcpyHostToDevice));

    const int blocks = (batch_count + KEM_ENCAPS_TPB - 1) / KEM_ENCAPS_TPB;
    batch_kem_encaps_serial_kernel<<<blocks, KEM_ENCAPS_TPB>>>(device_ciphertexts.get(),
                                                               device_secrets.get(),
                                                               device_public_keys.get(),
                                                               device_coins.get(),
                                                               batch_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ciphertext.data(),
                        device_ciphertexts.get(),
                        ciphertext.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(shared_secret.data(),
                        device_secrets.get(),
                        shared_secret.size(),
                        hipMemcpyDeviceToHost));

    if(write_file(ciphertext_path, ciphertext.data(), ciphertext.size()) != 0
       || write_file(secret_path, shared_secret.data(), shared_secret.size()) != 0)
        return 2;

    std::printf("KEM encapsulation PASS: %s\n", CRYPTO_ALGNAME);
    return 0;
}

int run_decaps(int argc, char** argv, int batch_count)
{
    const char* secret_key_path = argument_value(argc, argv, "--sk-in");
    const char* ciphertext_path = argument_value(argc, argv, "--ct-in");
    const char* secret_path     = argument_value(argc, argv, "--ss-out");
    if(secret_key_path == nullptr || ciphertext_path == nullptr || secret_path == nullptr)
    {
        std::fprintf(stderr, "Decapsulation requires --sk-in, --ct-in, and --ss-out.\n");
        return 2;
    }

    std::vector<uint8_t> secret_key;
    std::vector<uint8_t> ciphertext;
    if(read_file(secret_key_path, &secret_key, PARAM_SECRETKEYBYTES) != 0
       || read_file(ciphertext_path, &ciphertext, PARAM_CIPHERTEXTBYTES) != 0)
        return 2;

    std::vector<uint8_t> secret_keys;
    std::vector<uint8_t> ciphertexts;
    repeat_record(&secret_keys, secret_key, batch_count);
    repeat_record(&ciphertexts, ciphertext, batch_count);
    std::vector<uint8_t> shared_secret(PARAM_SSBYTES);

    DeviceBuffer<uint8_t> device_secret_keys;
    DeviceBuffer<uint8_t> device_ciphertexts;
    DeviceBuffer<uint8_t> device_secrets;
    HIP_CHECK(device_secret_keys.allocate(secret_keys.size()));
    HIP_CHECK(device_ciphertexts.allocate(ciphertexts.size()));
    HIP_CHECK(device_secrets.allocate(static_cast<std::size_t>(batch_count) * PARAM_SSBYTES));
    HIP_CHECK(hipMemcpy(device_secret_keys.get(),
                        secret_keys.data(),
                        secret_keys.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_ciphertexts.get(),
                        ciphertexts.data(),
                        ciphertexts.size(),
                        hipMemcpyHostToDevice));

    const int blocks = (batch_count + KEM_DECAPS_TPB - 1) / KEM_DECAPS_TPB;
    batch_kem_decaps_serial_kernel<<<blocks, KEM_DECAPS_TPB>>>(device_secrets.get(),
                                                               device_ciphertexts.get(),
                                                               device_secret_keys.get(),
                                                               batch_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(shared_secret.data(),
                        device_secrets.get(),
                        shared_secret.size(),
                        hipMemcpyDeviceToHost));

    if(write_file(secret_path, shared_secret.data(), shared_secret.size()) != 0)
        return 2;

    std::printf("KEM decapsulation PASS: %s\n", CRYPTO_ALGNAME);
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

    const hipError_t stack_result = hipDeviceSetLimit(hipLimitStackSize, 64u * 1024u);
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
        case Mode::keygen: return run_keygen(argc, argv, batch_count);
        case Mode::encaps: return run_encaps(argc, argv, batch_count);
        case Mode::decaps: return run_decaps(argc, argv, batch_count);
    }

    return 2;
}
