// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

// Paged or pinned host memory
enum class MemoryMode : unsigned int
{
    PAGED,
    PINNED
};

// Test either ranges of inputs sizes with a constant increament
// or a more complex shmoo test that tests bandwidth for large number of varying sizes.
enum class TestMode : unsigned int
{
    RANGED,
    SHMOO
};

/// \brief Run host to device or device to host transfer, bandwidth calculated for the specified configuration
std::vector<double>
    run_bandwidth_host_device(const std::vector<size_t>& memory_copy_measurement_sizes,
                              const int                  device,
                              hipMemcpyKind              hip_memcpy_kind,
                              const MemoryMode           memory_mode,
                              const unsigned int         trails)
{

    // Check for invalid configurations
    if(hip_memcpy_kind == hipMemcpyDeviceToDevice)
    {
        std::cerr << "hipMemcpyDeviceToDevice is an invalid Configuration\n";
        exit(error_exit_code);
    }

    // The bandwidths calculated will be stored in bandwidth_measurements
    std::vector<double> bandwidth_measurements;

    // Flush buffer for CPU cache
    constexpr size_t  flush_size = 256 * 1024 * 1024;
    std::vector<char> flush_buffer(flush_size);

    HIP_CHECK(hipSetDevice(device));

    if(hip_memcpy_kind == hipMemcpyHostToDevice)
    {
        std::cout << "Measuring Host to Device Bandwidth: " << std::flush;
    }
    else
    {
        std::cout << "Measuring Device to Host Bandwidth: " << std::flush;
    }

    for(auto size : memory_copy_measurement_sizes)
    {
        std::cout << "[" << size << "] " << std::flush;

        // Blocks used to clear host cache
        const unsigned long long   cache_clear_size = 1 << 24;
        std::vector<unsigned char> h_cache_block_1(cache_clear_size);
        std::vector<unsigned char> h_cache_block_2(cache_clear_size);

        // Size in bytes
        const size_t size_in_bytes = sizeof(unsigned char) * size;

        // Allocate device input memory
        unsigned char* d_in = nullptr;
        HIP_CHECK(hipMalloc(&d_in, size_in_bytes));

        // Memory transfer from host to device
        if(memory_mode == MemoryMode::PAGED)
        {
            // Host input memory
            std::vector<unsigned char> h_in(size);

            // Host output memory
            std::vector<unsigned char> h_out(size);

            // Initialize the host input memory
            for(unsigned int i = 0; i < size; i++)
            {
                h_in[i] = static_cast<unsigned char>(i & 0xff);
            }

            unsigned char* src = nullptr;
            unsigned char* dst = nullptr;

            switch(hip_memcpy_kind)
            {
                case hipMemcpyHostToDevice:
                    // Set the source and destination for hipMemcpy
                    src = h_in.data();
                    dst = d_in;
                    break;
                case hipMemcpyDeviceToHost:
                    // Transfer the host input to device
                    HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_in_bytes, hipMemcpyHostToDevice));

                    // Set the source and destination for hipMemcpy
                    src = d_in;
                    dst = h_out.data();
                    break;
                default:
                    std::cerr << "Invalid memcpy kind " << hip_memcpy_kind << "! \n";
                    exit(error_exit_code);
            }

            // Fill the host cache clear buffers
            for(unsigned int i = 0; i < h_cache_block_1.size(); i++)
            {
                h_cache_block_1[i] = static_cast<unsigned char>(i & 0xff);
                h_cache_block_2[i] = static_cast<unsigned char>(0xff - (i & 0xff));
            }

            // Timer class
            HostClock host_clock;

            // Perform memory transfers warm up
            for(unsigned int i = 0; i < 5; i++)
            {
                // Initiate the memory transfer
                HIP_CHECK(hipMemcpy(dst, src, size_in_bytes, hip_memcpy_kind));

                // Flush the buffer
                std::memset(flush_buffer.data(), i, flush_buffer.size());
            }

            // Perform memory transfers for trails number of times
            for(unsigned int i = 0; i < trails; i++)
            {
                host_clock.start_timer();

                // Initiate the memory transfer
                HIP_CHECK(hipMemcpy(dst, src, size_in_bytes, hip_memcpy_kind));

                host_clock.stop_timer();

                // Flush the buffer
                std::memset(flush_buffer.data(), i, flush_buffer.size());
            }
            // Calculate the bandwith in GB/s
            const double bandwidth_achieved
                = ((size_in_bytes * trails) / 1e9) / host_clock.get_elapsed_time();

            bandwidth_measurements.emplace_back(bandwidth_achieved);
        }
        else if(memory_mode == MemoryMode::PINNED) // Pinned memory mode
        {
            // Host input memory
            unsigned char* h_in = nullptr;

            // Host output memory
            unsigned char* h_out = nullptr;

            HIP_CHECK(hipHostMalloc(&h_in, size_in_bytes));
            HIP_CHECK(hipHostMalloc(&h_out, size_in_bytes));

            // Initialize the host memory
            for(unsigned int i = 0; i < size; i++)
            {
                h_in[i] = static_cast<unsigned char>(i & 0xff);
            }

            unsigned char* src = nullptr;
            unsigned char* dst = nullptr;

            if(hip_memcpy_kind == hipMemcpyHostToDevice)
            {
                // Set the source and destination for hipMemcpy
                src = h_in;
                dst = d_in;
            }
            else if(hip_memcpy_kind == hipMemcpyDeviceToHost)
            {
                // Transfer the host input to device
                HIP_CHECK(hipMemcpyAsync(d_in, h_in, size_in_bytes, hip_memcpy_kind));
                HIP_CHECK(hipDeviceSynchronize());

                // Set the source and destination for hipMemcpy
                src = d_in;
                dst = h_out;
            }

            // Perform memory transfers warm up
            for(unsigned int i = 0; i < 5; i++)
            {
                HIP_CHECK(hipMemcpyAsync(dst, src, size_in_bytes, hip_memcpy_kind));
            }
            HIP_CHECK(hipDeviceSynchronize());

            HostClock host_clock;
            host_clock.start_timer();

            // Initiate the memory transfer
            // Perform memory transfers for trails number of times
            for(unsigned int i = 0; i < trails; i++)
            {
                HIP_CHECK(hipMemcpyAsync(dst, src, size_in_bytes, hip_memcpy_kind));
            }

            HIP_CHECK(hipDeviceSynchronize());

            host_clock.stop_timer();

            // Calculate the bandwith in GB/s
            const double bandwidth_achieved
                = ((size_in_bytes * trails) / 1e9) / host_clock.get_elapsed_time();

            bandwidth_measurements.emplace_back(bandwidth_achieved);

            HIP_CHECK(hipHostFree(h_in));
            HIP_CHECK(hipHostFree(h_out));
        }

        // Free the memory
        HIP_CHECK(hipFree(d_in));
    }
    std::cout << std::endl;

    return bandwidth_measurements;
}

/// \brief Run device to device transfer, bandwidth calculated for the specified configuration
std::vector<double> run_bandwidth_device_device(std::vector<size_t> memory_copy_measurement_sizes,
                                                const int           device,
                                                const unsigned int  trails)
{

    // The bandwidths calculated will be stored in bandwidth_measurements
    std::vector<double> bandwidth_measurements;

    HIP_CHECK(hipSetDevice(device));

    std::cout << "Measuring Device to Device Bandwith: " << std::flush;
    for(auto size : memory_copy_measurement_sizes)
    {
        std::cout << "[" << size << "] " << std::flush;

        // Size in bytes
        const size_t size_in_bytes = sizeof(unsigned char) * size;

        // Allocate device input memory
        unsigned char* d_in = nullptr;
        HIP_CHECK(hipMalloc(&d_in, size_in_bytes));

        // Device output memory.
        unsigned char* d_out = nullptr;
        HIP_CHECK(hipMalloc(&d_out, size_in_bytes));

        // Host input memory
        std::vector<unsigned char> h_in(size);

        // Initialize the host input memory
        for(unsigned int i = 0; i < size; i++)
        {
            h_in[i] = static_cast<unsigned char>(i & 0xff);
        }

        // Transfer the host input to device
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_in_bytes, hipMemcpyHostToDevice));

        // Set the source and destination for hipMemcpy
        unsigned char* src = d_in;
        unsigned char* dst = d_out;

        // Perform memory transfers warm up
        for(unsigned int i = 0; i < 5; i++)
        {
            // Initiate the memory transfer
            HIP_CHECK(hipMemcpy(dst, src, size_in_bytes, hipMemcpyDeviceToDevice));
        }

        // Synchronize because the device to device memory copy is non-blocking
        HIP_CHECK(hipDeviceSynchronize());

        // Timer class
        HostClock host_clock;
        host_clock.start_timer();

        // Perform memory transfers for trails number of times
        for(unsigned int i = 0; i < trails; i++)
        {
            // Initiate the memory transfer
            HIP_CHECK(hipMemcpy(dst, src, size_in_bytes, hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipDeviceSynchronize());

        host_clock.stop_timer();

        // Calculate the bandwith in GB/s
        const double bandwidth_achieved
            = ((size_in_bytes * trails) / 1e9) / host_clock.get_elapsed_time();

        bandwidth_measurements.emplace_back(bandwidth_achieved);

        // Free the device output memory
        HIP_CHECK(hipFree(d_out));

        // Free the memory
        HIP_CHECK(hipFree(d_in));
    }
    std::cout << std::endl;

    return bandwidth_measurements;
}

std::vector<size_t> generate_measurement_sizes_range(const size_t start_measurement,
                                                     const size_t end_measurement,
                                                     const size_t stride_between_measurements)
{
    // The size of data to copy for each measurement
    std::vector<size_t> memory_copy_measurement_sizes;

    for(size_t i = start_measurement; i < end_measurement; i += stride_between_measurements)
    {
        memory_copy_measurement_sizes.emplace_back(i);
    }

    return memory_copy_measurement_sizes;
}

std::vector<size_t> generate_measurement_sizes_shmoo()
{

    // Constants for shmoo mode
    const size_t shmoo_memsize_max = 1 << 26; // 64 MB

    const size_t shmoo_increment_1KB   = 1 << 10; // 1 KB
    const size_t shmoo_increment_2KB   = 1 << 11; // 2 KB
    const size_t shmoo_increment_10KB  = shmoo_increment_1KB * 10; // 10KB
    const size_t shmoo_increment_100KB = shmoo_increment_10KB * 10; // 100 KB
    const size_t shmoo_increment_1MB   = 1 << 20; // 1 MB
    const size_t shmoo_increment_2MB   = 1 << 21; // 2 MB
    const size_t shmoo_increment_4MB   = 1 << 22; // 4 MB

    const size_t shmoo_limit_20KB  = shmoo_increment_10KB * 2; // 20 KB
    const size_t shmoo_limit_50KB  = shmoo_increment_10KB * 5; // 50 KB
    const size_t shmoo_limit_100KB = shmoo_increment_10KB * 10; // 100 KB
    const size_t shmoo_limit_1MB   = 1 << 20; // 1 MB
    const size_t shmoo_limit_16MB  = 1 << 24; // 16 MB
    const size_t shmoo_limit_32MB  = 1 << 25; // 32 MB

    // The size of data to copy for each measurement
    std::vector<size_t> memory_copy_measurement_sizes;

    size_t current_size = 0;

    while(current_size <= shmoo_memsize_max)
    {
        if(current_size < shmoo_limit_20KB)
        {
            current_size += shmoo_increment_1KB;
        }
        else if(current_size < shmoo_limit_50KB)
        {
            current_size += shmoo_increment_2KB;
        }
        else if(current_size < shmoo_limit_100KB)
        {
            current_size += shmoo_increment_10KB;
        }
        else if(current_size < shmoo_limit_1MB)
        {
            current_size += shmoo_increment_100KB;
        }
        else if(current_size < shmoo_limit_16MB)
        {
            current_size += shmoo_increment_1MB;
        }
        else if(current_size < shmoo_limit_32MB)
        {
            current_size += shmoo_increment_2MB;
        }
        else
        {
            current_size += shmoo_increment_4MB;
        }
        memory_copy_measurement_sizes.emplace_back(current_size);
    }

    return memory_copy_measurement_sizes;
}

void configure_parser(cli::Parser& parser)
{
    // Default parameters
    parser.set_optional<size_t>("start", "start", 1 << 20, "Starting size"); // Default 1 MB
    parser.set_optional<size_t>("end", "end", 1 << 23, "Ending size"); // Default 8 MB
    parser.set_optional<size_t>("stride",
                                "stride",
                                1 << 22, // Default 4 MB
                                "Stride (or increament) between sizes");

    parser.set_optional<std::string>("mode",
                                     "mode",
                                     "range",
                                     "Mode of bandwidth test: range or shmoo");
    parser.set_optional<std::string>("memory",
                                     "memory",
                                     "pageable",
                                     "Memory allocation kind: pageable or pinned\n");
    parser.set_optional<size_t>("trials", "trials", 50, "Number of trials");
    parser.set_optional<std::vector<std::string>>(
        "device",
        "device",
        {"0"},
        "Space-separated list of devices\n"
        "\tall for using all the available devices\n"
        "\t0,1,2,...,n for using any particular available devices");
    parser.set_optional<std::vector<std::string>>("memcpy",
                                                  "memcpy",
                                                  {"htod", "dtoh", "dtod"},
                                                  "Space-separated list of memory copy kind.\n"
                                                  "\thtod is host to device\n"
                                                  "\tdtoh is device to host\n"
                                                  "\tdtod is device to device");
}

int main(int argc, char** argv)
{

    // Get the number of hip devices in the system
    int number_of_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&number_of_devices))

    if(number_of_devices <= 0)
    {
        std::cerr << "HIP supported devices not found!"
                  << "\n";
        exit(error_exit_code);
    }

    // Parse user inputs
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Set configurations for testing bandwidth
    const size_t                   trials                      = parser.get<size_t>("trials");
    const size_t                   start_measurement           = parser.get<size_t>("start");
    const size_t                   end_measurement             = parser.get<size_t>("end");
    const size_t                   stride_between_measurements = parser.get<size_t>("stride");
    const std::string              mode                        = parser.get<std::string>("mode");
    const std::string              memory_cmd                  = parser.get<std::string>("memory");
    const std::vector<std::string> devices_cmd = parser.get<std::vector<std::string>>("device");
    const std::vector<std::string> memcpy_cmd  = parser.get<std::vector<std::string>>("memcpy");

    // Set the mode of bandwidth test: RANGED or SHMOO
    TestMode mode_of_test;

    if(mode == "range")
    {
        mode_of_test = TestMode::RANGED;
    }
    else if(mode == "shmoo")
    {
        mode_of_test = TestMode::SHMOO;
    }
    else
    {
        std::cerr << "Invalid mode " << mode << "! \n";
        exit(error_exit_code);
    }

    // Set the memory host allocation type: PAGED or PINNED
    MemoryMode memory_allocation;
    if(memory_cmd == "pageable")
    {
        memory_allocation = MemoryMode::PAGED;
    }
    else if(memory_cmd == "pinned")
    {
        memory_allocation = MemoryMode::PINNED;
    }
    else
    {
        std::cerr << "Invalid memory allocation " << memory_cmd << "! \n";
        exit(error_exit_code);
    }

    // Store device ids
    std::vector<int> devices;
    if(std::find(devices_cmd.begin(), devices_cmd.end(), "all") != devices_cmd.end())
    {
        devices = std::vector<int>(number_of_devices);

        // Initialize the default device ids
        std::iota(devices.begin(), devices.end(), 0);
    }
    else
    {
        for(const std::string& device : devices_cmd)
        {
            int device_id;
            if(!parse_int_string(device, device_id))
            {
                std::cerr << "Invalid device ID " << device << "!\n";
                exit(error_exit_code);
            }

            if(device_id < 0 || device_id >= number_of_devices)
            {
                std::cerr << "Invalid device id " << device << "!\n"
                          << "Device does not exist\n";
                exit(error_exit_code);
            }
            devices.emplace_back(device_id);
        }
    }

    std::cout << "Devices: " << format_range(devices.begin(), devices.end()) << "\n";

    // Set hipMemcpyKind
    std::map<hipMemcpyKind, std::string> memcpy_kinds;
    if(std::find(memcpy_cmd.begin(), memcpy_cmd.end(), "all") != memcpy_cmd.end())
    {
        memcpy_kinds.insert({hipMemcpyHostToDevice, "Host to Device"});
        memcpy_kinds.insert({hipMemcpyDeviceToHost, "Device to Host"});
        memcpy_kinds.insert({hipMemcpyDeviceToDevice, "Device to Device"});
    }
    else
    {
        for(std::string memcpy : memcpy_cmd)
        {
            if(memcpy == "htod")
            {
                memcpy_kinds.insert({hipMemcpyHostToDevice, "Host to Device"});
            }
            else if(memcpy == "dtoh")
            {
                memcpy_kinds.insert({hipMemcpyDeviceToHost, "Device to Host"});
            }
            else if(memcpy == "dtod")
            {
                memcpy_kinds.insert({hipMemcpyDeviceToDevice, "Device to Device"});
            }
            else
            {
                std::cerr << "Invalid memcpy!"
                          << "\n";
                exit(error_exit_code);
            }
        }
    }

    std::vector<size_t> memory_copy_measurement_sizes;
    if(mode_of_test == TestMode::RANGED)
    {
        memory_copy_measurement_sizes
            = generate_measurement_sizes_range(start_measurement,
                                               end_measurement,
                                               stride_between_measurements);
    }
    else
    {
        memory_copy_measurement_sizes = generate_measurement_sizes_shmoo();
    }

    std::cout << "Measurement Sizes: "
              << format_range(memory_copy_measurement_sizes.begin(),
                              memory_copy_measurement_sizes.end())
              << "\n\n";

    // Run the bandwidth tests on devices
    for(auto device : devices)
    {
        hipDeviceProp_t devProp;
        HIP_CHECK(hipSetDevice(device));
        HIP_CHECK(hipGetDeviceProperties(&devProp, device));

        for(auto memcpy_kind : memcpy_kinds)
        {
            std::string print_text;
            if(memory_allocation == MemoryMode::PAGED)
            {
                print_text = "Paged Bandwidth ";
            }
            else if(memory_allocation == MemoryMode::PINNED)
            {
                print_text = "Pinned Bandwidth ";
            }
            if(memcpy_kind.first == hipMemcpyDeviceToDevice)
            {
                print_text = "Bandwidth ";
            }

            std::vector<double> bandwidth_measurements;
            if(memcpy_kind.first == hipMemcpyDeviceToDevice)
            {
                bandwidth_measurements
                    = run_bandwidth_device_device(memory_copy_measurement_sizes, device, trials);
            }
            else
            {
                bandwidth_measurements = run_bandwidth_host_device(memory_copy_measurement_sizes,
                                                                   device,
                                                                   memcpy_kind.first,
                                                                   memory_allocation,
                                                                   trials);
            }
            std::cout << "\nDevice ID [" << device << "] Device Name [" << devProp.name
                      << "]: " << print_text << memcpy_kind.second << " (GB/s): "
                      << format_range(bandwidth_measurements.begin(), bandwidth_measurements.end())
                      << "\n\n";
        }
    }
}
