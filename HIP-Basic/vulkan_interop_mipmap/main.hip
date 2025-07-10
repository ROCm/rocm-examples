// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "vulkan_utils.hpp"

#include "nvidia_hip_fix.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

// Include the SPIR-V shader headers
#include "texture.frag.spv.h"
#include "texture.vert.spv.h"

// Currently hip external semaphores are not working under Linux when
// compiling for the AMD platform.
// TODO: Remove once this is implemented in hipamd.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/48.
#ifndef USE_EXTERNAL_SEMAPHORES
    #if defined(__HIP_PLATFORM_AMD__) && !defined(_WIN64)
        #define USE_EXTERNAL_SEMAPHORES 0
    #else
        #define USE_EXTERNAL_SEMAPHORES 1
    #endif
#endif

// Currently it seems like waiting on an external semaphore that is signaled
// from hip is not working under windows
#ifndef USE_SIGNAL_SEMAPHORE
    #if defined(__HIP_PLATFORM_AMD__) && defined(_WIN64)
        #define USE_SIGNAL_SEMAPHORE 0

    #else
        #define USE_SIGNAL_SEMAPHORE 1
    #endif
#endif

/// \brief The maximum number of frames that can be rendered at the same time. By
/// setting this value to more than one, we can allow the presentation engine to
/// draw the rendered frame to the monitor while we already render the next frame
/// in the background.
constexpr size_t max_frames_in_flight = 2;

/// \brief Time maximum time (in nanoseconds) that we are willing to wait on the next
/// image from the swapchain.
constexpr uint64_t frame_timeout = std::numeric_limits<uint64_t>::max();

/// \brief The number of triangles that the example's grid is in width.
constexpr uint32_t grid_width = 256;
/// \brief The number of triangles that the example's grid is in height.
constexpr uint32_t grid_height = 256;

/// \brief The Vulkan instance extensions required for sharing HIP- and Vulkan
/// types. \p VK_KHR_external_memory_capabilities is required to share buffers, and
/// \p VK_KHR_external_semaphore_capabilities is required to share semaphores.
/// \p VK_KHR_get_physical_device_properties2 is required for the other two, as well
/// as for querying the device's UUID.
constexpr const char* required_instance_extensions[] = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

/// \brief The general Vulkan extensions that a particular device needs to support in order
/// for it to be able to run this example.
/// \p VK_KHR_swapchain is required in order to draw to the example's window, and \p VK_KHR_external_memory
/// and \p VK_KHR_external_semaphore are required to share memory and semaphores respectively with HIP.
constexpr const char* required_device_extensions[]
    = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
       VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
       VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
#else
       VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
#endif

/// \brief This structure represents a device UUID, obtained either from Vulkan or
/// from HIP.
struct uuid
{
    uint8_t bytes[VK_UUID_SIZE];

    /// \brief This function fetches a Vulkan-compatible device UUID from a HIP device.
    ///
    /// The use of this function should actually be replaced by \p hipDeviceGetUuid. However,
    /// on AMD it returns a device UUID that is not compatible with that returned by Vulkan, and
    /// when compiling for NVIDIA it yields a linker error. For this reason we provide our own
    /// implementation that is compatible with both the Mesa (RADV) and AMD (AMDVLK) implementations
    /// of Vulkan on AMD, and call into the CUDA API directly when compiling for NVIDIA.
    static uuid get_hip_device_uuid(hipDevice_t device)
    {
#if defined(__HIP_PLATFORM_AMD__)
        // The value that hipDeviceGetUuid returns does not correspond with those returned
        // by mesa (see https://gitlab.freedesktop.org/mesa/mesa/-/blob/5cd3e395037250946ba2519600836341df02c8ca/src/amd/common/ac_gpu_info.c#L1366-1382)
        // and by xgl (see https://github.com/GPUOpen-Drivers/xgl/blob/4118707939c2f4783d28ce2a383184a3794ca477/icd/api/vk_physical_device.cpp#L4363-L4421)
        // Those drivers _do_ align with each other, so we can create our own UUID here.
        // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/50.
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device));

        struct uuid result    = {};
        uint32_t*   uuid_ints = reinterpret_cast<uint32_t*>(result.bytes);
        uuid_ints[0]          = props.pciDomainID;
        uuid_ints[1]          = props.pciBusID;
        uuid_ints[2]          = props.pciDeviceID;
        // Note: function is 0 anyway.

        return result;
#elif defined(__HIP_PLATFORM_NVIDIA__)
        // Work around a compile error related to hipDeviceGetUuid when compiling for NVIDIA:
        // "undefined reference to `cuDeviceGetUuid'"
        // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/51.
        cudaDeviceProp props;
        HIP_CHECK(hipCUDAErrorTohipError(cudaGetDeviceProperties(&props, device)));

        struct uuid result = {};
        std::memcpy(result.bytes, props.uuid.bytes, VK_UUID_SIZE);

        return result;
#else
    #error unsupported platform
#endif
    }
};

/// \brief \p std::ostream print operator overload for \p uuid.
/// \see uuid.
std::ostream& operator<<(std::ostream& os, const uuid uuid)
{
    for(size_t i = 0; i < VK_UUID_SIZE * 2; ++i)
    {
        // Extract the current nibble.
        const uint8_t c = (uuid.bytes[i / 2] >> (4 - (i % 2) * 4)) & 0xF;
        os << static_cast<char>(c < 10 ? c + '0' : c + 'a' - 10);
        if(i == 8 || i == 12 || i == 16 || i == 20)
        {
            os << '-';
        }
    }
    return os;
}

/// \brief This structure represents a candidate HIP-device that we can use
/// for this example.
struct hip_device_candidate
{
    /// The HIP device index representing this device.
    hipDevice_t device;
    /// The Vulkan-compatible device UUID.
    uuid device_uuid;
};

/// \brief This structure represents a candidate device that we can use for this
/// example.
struct physical_device_candidate
{
    /// The Vulkan physical device handle of the device to be used.
    VkPhysicalDevice pdev;

    /// The candidate device's Vulkan device properties.
    VkPhysicalDeviceProperties props;

    /// The HIP device candidate that this Vulkan device corresponds to.
    hip_device_candidate hip_candidate;

    /// The queue allocation that contains details about which queues will be
    /// used throughout this example.
    queue_allocation queues;
};

/// \brief Checks if a particular Vulkan physical device is qualified to run this example:
/// - It needs to support the Vulkan surface which we want to render to.
/// - It needs to support the required generic and platform-specific Vulkan device extensions.
/// - It needs to be a HIP-supported device. This is checked by fetching the device
///   UUID from Vulkan, and checking if it appears in the device UUIDs fetched from HIP
///   (passed through \p hip_uuids).
/// - It needs to support graphics- and present queues that can render to the surface.
/// If all of these are satisfied, the \p candidate structure is filled with information
/// about the physical device that is required later, and the function returns \p true.
/// Otherwise, \p false is returned.
///
/// \param hip_devices - A vector of \p hipDevice_t and their corresponding Vulkan-compatible
///   device UUID.
/// \param pdev - The Vulkan physical device to check suitability off.
/// \p surface - The Vulkan surface that the physical device needs to support.
bool is_physical_device_suitable(const instance_dispatch&                dispatch,
                                 const std::vector<hip_device_candidate> hip_devices,
                                 VkPhysicalDevice                        pdev,
                                 VkSurfaceKHR                            surface,
                                 physical_device_candidate&              candidate)
{
    // Check if HIP supports this device by checking if there is any device with the same UUID.
    {
        // Query the Vulkan device UUID using vkGetPhysicalDeviceProperties2.
        VkPhysicalDeviceIDPropertiesKHR id_props = {};
        id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2KHR props2 = {};
        props2.sType                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        props2.pNext                          = &id_props;

        dispatch.get_physical_device_properties2(pdev, &props2);

        const auto cmp_device_uuid = [&](const hip_device_candidate& hip_candidate)
        {
            return std::equal(std::begin(hip_candidate.device_uuid.bytes),
                              std::end(hip_candidate.device_uuid.bytes),
                              std::begin(id_props.deviceUUID),
                              std::end(id_props.deviceUUID));
        };

        // Try to find a HIP device UUID that matches the UUID reported by Vulkan - if any such exists,
        // we know that the device supports both Vulkan and HIP, and we can use it to run this example.
        const auto it = std::find_if(hip_devices.begin(), hip_devices.end(), cmp_device_uuid);
        if(it == hip_devices.end())
        {
            // This device does not support HIP.
            return false;
        }

        candidate.props         = props2.properties;
        candidate.hip_candidate = *it;
    }

    // Check if the device supports our surface at all.
    if(!check_surface_support(dispatch, pdev, surface))
    {
        return false;
    }

    // Check if the device supports the required extensions.
    if(!check_device_extensions(dispatch,
                                pdev,
                                required_device_extensions,
                                std::size(required_device_extensions)))
    {
        return false;
    }

    // Try to allocate device queues for the candidate device.
    if(!allocate_device_queues(dispatch, pdev, surface, candidate.queues))
    {
        return false;
    }

    candidate.pdev = pdev;
    return true;
}

/// \brief Try to find a physical device that can run this example. This is done by fetching
/// all supported devices from HIP and from Vulkan, and checking each of these to see if the required
/// features are supported.
///
/// To check whether a Vulkan and HIP device are the same, their UUIDs are compared.
/// \see \p uuid::get_hip_device_uuid.
/// \see \p is_physical_device_suitable.
void find_physical_device(const instance_dispatch&   dispatch,
                          VkInstance                 instance,
                          VkSurfaceKHR               surface,
                          physical_device_candidate& candidate)
{
    uint32_t physical_device_count;
    VK_CHECK(dispatch.enumerate_physical_devices(instance, &physical_device_count, nullptr));
    if(physical_device_count == 0)
    {
        std::cerr << "System has no physical devices\n";
        std::exit(error_exit_code);
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(dispatch.enumerate_physical_devices(instance,
                                                 &physical_device_count,
                                                 physical_devices.data()));

    // Fetch the number of HIP devices that are currently present on the system.
    // Note: This depends on the current HIP platform, and may report different
    // devices depending on that.
    int hip_device_count;
    HIP_CHECK(hipGetDeviceCount(&hip_device_count));
    std::vector<hip_device_candidate> hip_devices;
    for(hipDevice_t hip_device = 0; hip_device < hip_device_count; ++hip_device)
    {
        hipDeviceProp_t hip_properties;
        HIP_CHECK(hipGetDeviceProperties(&hip_properties, hip_device));
        if(hip_properties.computeMode != hipComputeModeProhibited)
        {
            hip_devices.push_back({hip_device, uuid::get_hip_device_uuid(hip_device)});
        }
    }

    for(VkPhysicalDevice pdev : physical_devices)
    {
        if(is_physical_device_suitable(dispatch, hip_devices, pdev, surface, candidate))
            return;
    }

    std::cerr << "No suitable device found\n";
    std::exit(error_exit_code);
}

/// \brief Allocate and bind memory for a Vulkan buffer
/// \param buffer - The buffer to allocate create memory for.
/// \param properties - The memory properties for the allocated memory.
/// \param external - Whether to allocate this memory such that it can be exported.
VkDeviceMemory allocate_buffer_memory(const graphics_context&     ctx,
                                      const VkBuffer              buffer,
                                      const VkMemoryPropertyFlags properties,
                                      const bool                  external = false)
{
    VkMemoryRequirements mem_reqs;
    ctx.vkd->get_buffer_memory_requirements(ctx.dev, buffer, &mem_reqs);
    const uint32_t memory_type = ctx.find_memory_type_index(mem_reqs.memoryTypeBits, properties);

    VkMemoryAllocateInfo allocate_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate_info.allocationSize       = mem_reqs.size;
    allocate_info.memoryTypeIndex      = memory_type;

    VkExportMemoryAllocateInfoKHR export_info = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR};
    if(external)
    {
#ifdef _WIN64
        export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
        export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        allocate_info.pNext = &export_info;
    }

    VkDeviceMemory memory;
    VK_CHECK(ctx.vkd->allocate_memory(ctx.dev, &allocate_info, nullptr, &memory));
    VK_CHECK(ctx.vkd->bind_buffer_memory(ctx.dev, buffer, memory, 0));
    return memory;
}

/// \brief Create and allocate a Vulkan buffer.
/// \param size - The size (in bytes) that this buffer should be allocated for.
/// \param usage - The Vulkan usage that this buffer will be used for.
/// \param external - If true, this buffer will be created so that it can later be exported to a
///   platform-native handle, that may be imported to HIP.
VkBuffer create_buffer(const graphics_context&  ctx,
                       const VkDeviceSize       size,
                       const VkBufferUsageFlags usage,
                       const bool               external = false)
{
    VkBufferCreateInfo create_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create_info.size               = size;
    create_info.usage              = usage;
    create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

    // In order to be able to export the buffer handle, we need to supply Vulkan with this
    // VkExternalMemoryBufferCreateInfoKHR, and set the handleTypes to the native handle type
    // that we want to export. Which handle type to export depends on the platform we are
    // currently compiling for.
    VkExternalMemoryBufferCreateInfoKHR external_create_info
        = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR};
    if(external)
    {
#ifdef _WIN64
        external_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
        external_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        create_info.pNext = &external_create_info;
    }

    VkBuffer buffer;
    VK_CHECK(ctx.vkd->create_buffer(ctx.dev, &create_info, nullptr, &buffer));
    return buffer;
}

/// \brief This function converts a Vulkan memory handle to its equivalent HIP handle. The
/// VkDeviceMemory passed to this function and the returned HIP memory represents the same
/// physical area of GPU memory, through the handles of each respective API. Writing to the
/// buffer in one API will allow us to read the results through the other. Note that access
/// to the buffer should be synchronized between the APIs, for example using queue syncs or
/// semaphores.
/// \param memory - The Vulkan memory handle to convert. This memory needs to be created with
///   the appropriate fields set in VkExportMemoryAllocateInfoKHR.
///   \see allocate_buffer_memory for allocating such a memory handle, and
///   \see create_buffer for creating a Vulkan buffer that is compatible with that memory.
hipExternalMemory_t
    memory_to_hip(const graphics_context& ctx, const VkDeviceMemory memory, const VkDeviceSize size)
{
    // Prepare the HIP external semaphore descriptor with the platform-specific
    // handle type that we wish to import. This value should correspond to the
    // handleTypes field set in VkExportMemoryAllocateInfoKHR while creating the
    // Vulkan buffer.
    hipExternalMemoryHandleDesc desc = {};
    desc.size                        = size;

    // Export the Vulkan buffer handle to a platform-specific native handle, depending
    // on the current platform: On Windows the buffer is converted to a HANDLE, and on Linux
    // to a file descriptor representing the driver's GPU handle to the memory.
    // This native handle is then passed to the HIP external memory descriptor so that it
    // may be imported.
#ifdef _WIN64
    desc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
    VkMemoryGetWin32HandleInfoKHR get_handle_info
        = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    get_handle_info.memory     = memory;
    get_handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
    VK_CHECK(
        ctx.vkd->get_memory_win32_handle(ctx.dev, &get_handle_info, &desc.handle.win32.handle));
#else
    desc.type                        = hipExternalMemoryHandleTypeOpaqueFd;
    VkMemoryGetFdInfoKHR get_fd_info = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    get_fd_info.memory               = memory;
    get_fd_info.handleType           = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    VK_CHECK(ctx.vkd->get_memory_fd(ctx.dev, &get_fd_info, &desc.handle.fd));
#endif

    // Import the native memory handle to HIP to create an external memory.
    hipExternalMemory_t hip_memory;
    HIP_CHECK(hipImportExternalMemory(&hip_memory, &desc));
    return hip_memory;
}

/// \brief Utility function to create a Vulkan semaphore.
/// \param external - If true, this semaphore is created so that it can later be exported
///   to a platform-native handle, which may be imported to HIP later.
VkSemaphore create_semaphore(const graphics_context& ctx, const bool external = false)
{
    VkSemaphoreCreateInfo create_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    // Similar to buffers, in order to be able to export the semaphore handle we need to supply
    // Vulkan with this VkExportSemaphoreCreateInfoKHR structure, and set the handleTypes to the
    // value appropriate for the platform that we are currently compiling for.
    VkExportSemaphoreCreateInfoKHR export_create_info
        = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR};
    if(external)
    {
#ifdef _WIN64
        export_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
        export_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        create_info.pNext = &export_create_info;
    }

    VkSemaphore sema;
    VK_CHECK(ctx.vkd->create_semaphore(ctx.dev, &create_info, nullptr, &sema));
    return sema;
}

/// \brief This function converts a Vulkan semaphore to its equivalent HIP handle. The passed
/// semaphore and the returned HIP semaphore represent the same backing semaphore, though the
/// handles of the respective API. Signaling on the semaphore in one API will allow the other
/// API to wait on it, which is how we can guarantee synchronized access to resources in a
/// cross-API manner.
/// \param sema - The Vulkan semaphore to convert. This semaphore needs to be created with
///   \p the appropriate fields set in VkExportSemaphoreCreateInfoKHR.
///   \see create_semaphore for creating such a semaphore.
hipExternalSemaphore_t semaphore_to_hip(const graphics_context& ctx, const VkSemaphore sema)
{
    // Prepare the HIP external semaphore descriptor with the platform-specific handle type
    // that we wish to import. This value should correspond to the handleTypes field set in
    // the VkExportSemaphoreCreateInfoKHR structure that was passed to Vulkan when creating
    // the semaphore.
    hipExternalSemaphoreHandleDesc desc = {};

    // Export the Vulkan semaphore to a platform-specific handle depending on the current
    // platform: On Windows, we convert the semaphore into a HANDLE, and on Linux it is
    // converted to a file descriptor.
    // This native handle is then passed to the HIP external semaphore descriptor.
#ifdef _WIN64
    desc.type = hipExternalSemaphoreHandleTypeOpaqueWin32;

    VkSemaphoreGetWin32HandleInfoKHR get_handle_info
        = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    get_handle_info.semaphore  = sema;
    get_handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    VK_CHECK(
        ctx.vkd->get_semaphore_win32_handle(ctx.dev, &get_handle_info, &desc.handle.win32.handle));
#else
    desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;

    VkSemaphoreGetFdInfoKHR get_fd_info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    get_fd_info.semaphore               = sema;
    get_fd_info.handleType              = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VK_CHECK(ctx.vkd->get_semaphore_fd(ctx.dev, &get_fd_info, &desc.handle.fd));
#endif

    // Import the native semaphore to HIP to create a HIP external semaphore.
    hipExternalSemaphore_t hip_sema;
    HIP_CHECK(hipImportExternalSemaphore(&hip_sema, &desc));
    return hip_sema;
}

// [Sphinx tunnel kernel start]
/// \brief Helper function that clamps a value between a minimum and maximum.
/// \param x - The value to clamp
/// \param min_val - The minimum allowed value
/// \param max_val - The maximum allowed value
/// \return The clamped value between min_val and max_val
__device__ inline float clampf(float x, float min_val, float max_val)
{
    return fminf(max_val, fmaxf(min_val, x));
}

/// \brief Helper function that performs a smooth interpolation between two values.
/// \param edge0 - The lower edge of the transition
/// \param edge1 - The upper edge of the transition
/// \param x - The input value to interpolate
/// \return A value smoothly interpolated between 0 and 1 when x is between edge0 and edge1
__device__ inline float smoothstep(float edge0, float edge1, float x)
{
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// \brief Helper function that returns the fractional part of a floating point number.
/// \param x - The input floating point number
/// \return The fractional part of x (the part after the decimal point)
__device__ inline float fracf(float x)
{
    return x - floorf(x);
}

/// \brief The main HIP kernel for this example - generates a tunnel-like effect
/// with dynamic texture data for Vulkan to render.
/// \param surface - HIP surface object to write RGBA pixel data to
/// \param width - Width of the texture in pixels
/// \param height - Height of the texture in pixels
/// \param time - Current animation time in seconds
/// \param speed - Animation speed for forward motion
/// \param pattern_freq - Frequency of pattern repetition
/// \param twist - Twist factor for spiral effect
__global__ void tunnel_kernel(hipSurfaceObject_t surface,
                              unsigned int       width,
                              unsigned int       height,
                              float              time,
                              float              speed,
                              float              pattern_freq,
                              float              twist)
{
    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < width && iy < height)
    {
        // Map pixel coordinates to normalized coordinates [-1, 1] centered at (0,0)
        float aspect_ratio = (float)height / (float)width;
        float u            = (2.0f * ix / (width - 1.0f) - 1.0f);
        float v            = (2.0f * iy / (height - 1.0f) - 1.0f) * aspect_ratio;

        // Convert to polar coordinates (angle, radius)
        float radius = sqrtf(u * u + v * v) + 1e-5f;
        float angle  = atan2f(v, u);

        // Map screen radius to a pseudo-depth coordinate
        float z = 1.0f / radius;

        // Animate the depth coordinate to simulate forward motion
        float animated_z = z + time * speed;

        // Define the repeating pattern based on angle and animated depth
        float angle_norm    = angle / (2.0f * M_PI);
        float twisted_angle = angle_norm + animated_z * twist;

        // Calculate pattern value based on depth and twisted angle
        float pattern_value = sinf(animated_z * pattern_freq * M_PI * 2.0f * 1.0f)
                              + cosf(twisted_angle * M_PI * 2.0f * 8.0f);

        // Determine Color based on pattern value with phase-shifted sine waves
        float color_freq = 1.0f;
        float r          = 0.5f + 0.5f * sinf(pattern_value * M_PI * color_freq + 0.0f);
        float g = 0.5f + 0.5f * sinf(pattern_value * M_PI * color_freq + 2.0f * M_PI / 3.0f);
        float b = 0.5f + 0.5f * sinf(pattern_value * M_PI * color_freq + 4.0f * M_PI / 3.0f);

        // Apply Depth Cueing (Fade to Black/Fog towards center)
        float fade_radius_start = 0.01f;
        float fade_radius_end   = 0.4f;
        float intensity         = smoothstep(fade_radius_start, fade_radius_end, radius);

        r *= intensity;
        g *= intensity;
        b *= intensity;

        // Clamp final colors
        r       = clampf(r, 0.0f, 1.0f);
        g       = clampf(g, 0.0f, 1.0f);
        b       = clampf(b, 0.0f, 1.0f);
        float a = 1.0f;

        float4 value = make_float4(r, g, b, a);

        // Write float4 value to surface
        surf2Dwrite(value, surface, ix * sizeof(float4), iy);
    }
}
// [Sphinx tunnel kernel end]

/// \brief In order to increase efficiency, we pipeline the rendering process. This allows us to render
/// the next frame already while another frame is being presented by Vulkan. The \p frame structure
/// contains the relevant Vulkan handles that are duplicated for each phase of the pipeline.
struct frame
{
    const graphics_context& ctx;

    /// The semaphore that guards the use of the swapchain image before it is ready.
    VkSemaphore image_acquired;
    /// The semaphore that guards the present before the image is rendered.
    VkSemaphore render_finished;
    /// A fence that allows us to synchronize on CPU until this frame is ready
    /// to be re-rendered again after it has been submitted to the GPU.
    VkFence frame_fence;
    /// The command pool that the command buffer for this frame will is allocated from.
    /// By having a separate pool for each frame we can reset the command for the frame simply
    /// by resetting the pool.
    VkCommandPool cmd_pool;
    /// The main command buffer for this frame.
    VkCommandBuffer cmd_buf;

    /// \brief Create a new <tt>frame</tt>.
    explicit frame(const graphics_context& ctx) : ctx(ctx)
    {
        this->image_acquired  = create_semaphore(ctx);
        this->render_finished = create_semaphore(ctx);

        VkFenceCreateInfo fence_create_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(ctx.vkd->create_fence(ctx.dev, &fence_create_info, nullptr, &this->frame_fence));

        VkCommandPoolCreateInfo cmd_pool_create_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cmd_pool_create_info.queueFamilyIndex        = ctx.graphics_queue.family;
        VK_CHECK(
            ctx.vkd->create_command_pool(ctx.dev, &cmd_pool_create_info, nullptr, &this->cmd_pool));

        VkCommandBufferAllocateInfo cmd_buf_allocate_info
            = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmd_buf_allocate_info.commandPool        = this->cmd_pool;
        cmd_buf_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_buf_allocate_info.commandBufferCount = 1;
        VK_CHECK(
            ctx.vkd->allocate_command_buffers(ctx.dev, &cmd_buf_allocate_info, &this->cmd_buf));
    }

    ~frame()
    {
        this->ctx.vkd->destroy_command_pool(this->ctx.dev, this->cmd_pool, nullptr);
        this->ctx.vkd->destroy_fence(this->ctx.dev, this->frame_fence, nullptr);
        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->image_acquired, nullptr);
        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->render_finished, nullptr);
    }

    /// \brief Wait until the GPU-work for this frame has been completed, so that we
    /// can render to it again.
    void wait() const
    {
        VK_CHECK(this->ctx.vkd->wait_for_fences(this->ctx.dev,
                                                1,
                                                &this->frame_fence,
                                                VK_TRUE,
                                                frame_timeout));
    }

    /// \brief Reset the fence that backs this frame.
    void reset() const
    {
        VK_CHECK(this->ctx.vkd->reset_fences(this->ctx.dev, 1, &this->frame_fence));
    }
};

/// \brief This structure contains all the rendering related information for this example.
/// Its contents differ itself from the \p graphics_context in that in a typical Vulkan programs
/// there is usually only one <tt>graphics_context<tt>-like structure, but there may be multiple
/// <tt>renderer</tt>-like structures. In this example though, there is only one.
///
/// This renderer renders a quad with a texture that is dynamically updated by HIP.
/// The texture is created in Vulkan and shared with HIP, allowing the HIP kernel
/// to write directly to the texture that will be displayed.
struct renderer
{
    /// The total number of vertices for the triangles.
    constexpr static size_t num_verts = grid_width * grid_height;
    /// The number of bytes in the x- and y-coordinates buffer. Each x/y coordinate is encoded as
    /// a pair of floats, which are stored in a packed array-of-structures format: | x | y | x | y | ... |.
    constexpr static size_t grid_buffer_size = num_verts * sizeof(float) * 2;
    /// The number of indices in the index buffer. Each triangle has 3 points, each square in the grid
    /// is made up of 2 triangles. There are (width - 1) by (height - 1) squares in the grid.
    constexpr static size_t num_indices = (grid_width - 1) * (grid_height - 1) * 3 * 2;
    /// The number of bytes in the index buffer. Each index is encoded as a 32-bit int.
    constexpr static size_t index_buffer_size = num_indices * sizeof(uint32_t);

    const graphics_context& ctx;
    swapchain&              sc;
    hipDevice_t             hip_device;
    hipStream_t             hip_stream;

    /// The Vulkan render pass for rendering to the swapchain.
    VkRenderPass render_pass;
    /// The frames in the rendering pipeline.
    std::vector<frame> frames;
    /// The index of the frame we are currently rendering to.
    uint32_t frame_index = 0;
    /// The Vulkan frame buffers to render to - each corresponds to a swapchain
    /// image with the same index in <tt>sc</tt>
    std::vector<VkFramebuffer> framebuffers;

    /// The pipeline layout and pipeline of the rendering pipeline for the Vulkan part
    /// of this example.
    VkPipelineLayout pipeline_layout;
    VkPipeline       pipeline;
    /// Whether the swapchain is out-of-date and needs to be recreated.
    bool swapchain_out_of_date = false;

    // Grid and Index Buffers for geometry
    /// The buffer and memory holding the grid coordinates.
    VkBuffer       grid_buffer;
    VkDeviceMemory grid_memory;
    /// The buffer and memory holding the indices for the triangles to render.
    VkBuffer       index_buffer;
    VkDeviceMemory index_memory;

    // Shared Image Resources (Optimal Tiling, RGBA32F)
    /// The image that will be shared between Vulkan and HIP.
    VkImage height_image;
    /// The memory backing the shared image.
    VkDeviceMemory height_image_memory;
    /// The image view for accessing the shared image in shaders.
    VkImageView height_image_view;
    /// The sampler for sampling the shared image in the fragment shader.
    VkSampler texture_sampler;

    // HIP Interop Handles (for the shared image memory)
    /// The HIP external memory handle for the shared image.
    hipExternalMemory_t hip_image_memory_handle;
    /// The HIP mipmapped array structure for the shared image.
    hipMipmappedArray_t hip_height_mipmapped_array;
    /// The HIP surface object used to write to the shared image.
    hipSurfaceObject_t hip_height_surface;

    // Descriptor Set Resources for Texture Access
    /// The descriptor set layout for the texture sampler.
    VkDescriptorSetLayout descriptor_set_layout;
    /// The descriptor pool for allocating descriptor sets.
    VkDescriptorPool descriptor_pool;
    /// The descriptor set that references the shared image.
    VkDescriptorSet descriptor_set;

    // Synchronization Semaphores for Vulkan-HIP interaction
    /// The semaphore that signals when an image is ready for HIP processing.
    VkSemaphore image_ready_for_hip;
    /// The HIP-imported version of the image_ready_for_hip semaphore.
    hipExternalSemaphore_t hip_image_ready_for_hip;
    /// The semaphore that signals when HIP simulation is finished.
    VkSemaphore simulation_finished_for_vk;
    /// The HIP-imported version of the simulation_finished_for_vk semaphore.
    hipExternalSemaphore_t hip_simulation_finished_for_vk;

    // Performance measurement and timing variables
    /// The time at which this example started.
    std::chrono::high_resolution_clock::time_point start_time;
    /// Frame counter for FPS calculation.
    uint32_t fps_start_frame = 0;
    /// Timestamp of when we started measuring FPS.
    std::chrono::high_resolution_clock::time_point fps_start_time;

    /// \brief Initialize the renderer with all necessary Vulkan and HIP resources.
    /// \param ctx - The graphics context providing Vulkan function pointers.
    /// \param sc - The swapchain to which the renderer will render.
    /// \param hip_device - The HIP device index to use for computation.
    renderer(const graphics_context& ctx, swapchain& sc, const hipDevice_t hip_device)
        : ctx(ctx), sc(sc), hip_device(hip_device)
    {
        // Create a HIP stream for the specified device
        HIP_CHECK(hipSetDevice(this->hip_device));
        HIP_CHECK(hipStreamCreate(&this->hip_stream));

        // Initialize the Vulkan resources
        this->render_pass = sc.create_render_pass();
        this->create_grid_and_index_buffers();
        this->initialize_buffer_data();
        this->create_texture_resources();
        this->create_pipeline_layout();
        this->create_pipeline();
        this->initialize_hip_interop_resources();
        this->create_interop_semaphores();

        // Setup rendering pipeline frames
        this->frames.reserve(max_frames_in_flight);
        for(size_t i = 0; i < max_frames_in_flight; ++i)
        {
            this->frames.emplace_back(ctx);
        }
        this->sc.recreate_framebuffers(this->render_pass, this->framebuffers);

        // Initialize timing information for performance measurement
        this->start_time     = std::chrono::high_resolution_clock::now();
        this->fps_start_time = this->start_time;
    }

    /// \brief Clean up all resources used by the renderer.
    ~renderer()
    {
        // Wait for all rendering and compute operations to complete
        VK_CHECK(this->ctx.vkd->device_wait_idle(this->ctx.dev));
        if(this->hip_stream)
        {
            HIP_CHECK(hipStreamSynchronize(this->hip_stream));
            HIP_CHECK(hipStreamDestroy(this->hip_stream));
            this->hip_stream = nullptr; // Avoid double destroy if called again
        }

        // Destroy HIP-Vulkan synchronization resources
#if USE_EXTERNAL_SEMAPHORES == 1
        if(this->hip_image_ready_for_hip)
            HIP_CHECK(hipDestroyExternalSemaphore(this->hip_image_ready_for_hip));
        if(this->hip_simulation_finished_for_vk)
            HIP_CHECK(hipDestroyExternalSemaphore(this->hip_simulation_finished_for_vk));
        if(this->image_ready_for_hip)
            this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->image_ready_for_hip, nullptr);
        if(this->simulation_finished_for_vk)
            this->ctx.vkd->destroy_semaphore(this->ctx.dev,
                                             this->simulation_finished_for_vk,
                                             nullptr);
#endif

        // Destroy HIP texture resources
        if(this->hip_height_surface)
            HIP_CHECK(hipDestroySurfaceObject(this->hip_height_surface));
        if(this->hip_height_mipmapped_array)
            HIP_CHECK(hipFreeMipmappedArray(this->hip_height_mipmapped_array));
        if(this->hip_image_memory_handle)
            HIP_CHECK(hipDestroyExternalMemory(this->hip_image_memory_handle));

        // Destroy Vulkan descriptor resources
        if(this->descriptor_pool)
            this->ctx.vkd->destroy_descriptor_pool(this->ctx.dev, this->descriptor_pool, nullptr);
        if(this->descriptor_set_layout)
            this->ctx.vkd->destroy_descriptor_set_layout(this->ctx.dev,
                                                         this->descriptor_set_layout,
                                                         nullptr);

        // Destroy Vulkan texture resources
        if(this->texture_sampler)
            this->ctx.vkd->destroy_sampler(this->ctx.dev, this->texture_sampler, nullptr);
        if(this->height_image_view)
            this->ctx.vkd->destroy_image_view(this->ctx.dev, this->height_image_view, nullptr);
        if(this->height_image)
            this->ctx.vkd->destroy_image(this->ctx.dev, this->height_image, nullptr);
        if(this->height_image_memory)
            this->ctx.vkd->free_memory(this->ctx.dev, this->height_image_memory, nullptr);

        // Destroy Vulkan buffer resources
        if(this->index_memory)
            this->ctx.vkd->free_memory(this->ctx.dev, this->index_memory, nullptr);
        if(this->grid_memory)
            this->ctx.vkd->free_memory(this->ctx.dev, this->grid_memory, nullptr);
        if(this->index_buffer)
            this->ctx.vkd->destroy_buffer(this->ctx.dev, this->index_buffer, nullptr);
        if(this->grid_buffer)
            this->ctx.vkd->destroy_buffer(this->ctx.dev, this->grid_buffer, nullptr);

        // Destroy Vulkan pipeline resources
        if(this->pipeline)
            this->ctx.vkd->destroy_pipeline(this->ctx.dev, this->pipeline, nullptr);
        if(this->pipeline_layout)
            this->ctx.vkd->destroy_pipeline_layout(this->ctx.dev, this->pipeline_layout, nullptr);

        // Destroy framebuffers
        for(const VkFramebuffer fb : this->framebuffers)
        {
            if(fb)
                this->ctx.vkd->destroy_framebuffer(this->ctx.dev, fb, nullptr);
        }

        // Destroy render pass
        if(this->render_pass)
            this->ctx.vkd->destroy_render_pass(this->ctx.dev, this->render_pass, nullptr);
        // Frame destructors handle frame cleanup automatically
    }

    // Delete copy/move operations
    renderer(const renderer&)            = delete;
    renderer& operator=(const renderer&) = delete;
    renderer(renderer&&)                 = delete;
    renderer& operator=(renderer&&)      = delete;

    /// \brief Block until all current frames have finished rendering.
    void wait_all_frames()
    {
        VK_CHECK(this->ctx.vkd->device_wait_idle(this->ctx.dev));
    }

    /// \brief Create vertex and index buffers for the grid geometry.
    void create_grid_and_index_buffers()
    {
        // Create a buffer for vertex positions
        this->grid_buffer
            = create_buffer(ctx,
                            grid_buffer_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        this->grid_memory
            = allocate_buffer_memory(ctx, this->grid_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Create a buffer for triangle indices
        this->index_buffer
            = create_buffer(ctx,
                            index_buffer_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        this->index_memory
            = allocate_buffer_memory(ctx, this->index_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    /// \brief Initialize the grid and index buffer data.
    void initialize_buffer_data()
    {
        // Create a staging buffer for transferring data to device-local memory
        constexpr size_t staging_buffer_size = std::max(grid_buffer_size, index_buffer_size);
        VkBuffer         staging_buffer
            = create_buffer(ctx, staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkDeviceMemory staging_memory = allocate_buffer_memory(
            ctx,
            staging_buffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Map the staging buffer into host memory
        void* staging;
        VK_CHECK(
            this->ctx.vkd
                ->map_memory(this->ctx.dev, staging_memory, 0, staging_buffer_size, 0, &staging));

        // Initialize the grid buffer with normalized vertex positions
        float* grid = reinterpret_cast<float*>(staging);
        for(uint32_t y = 0; y < grid_height; ++y)
        {
            for(uint32_t x = 0; x < grid_width; ++x)
            {
                *grid++ = (2.0f * x) / (grid_width - 1) - 1.0f;
                *grid++ = (2.0f * y) / (grid_height - 1) - 1.0f;
            }
        }
        this->ctx.copy_buffer(this->grid_buffer, staging_buffer, grid_buffer_size);

        // Initialize the index buffer for triangle connectivity
        uint32_t* indices = reinterpret_cast<uint32_t*>(staging);
        for(uint32_t y = 0; y < grid_height - 1; ++y)
        {
            for(uint32_t x = 0; x < grid_width - 1; ++x)
            {
                // First triangle (top-left, bottom-left, top-right)
                *indices++ = (y + 0) * grid_width + (x + 0);
                *indices++ = (y + 1) * grid_width + (x + 0);
                *indices++ = (y + 0) * grid_width + (x + 1);

                // Second triangle (top-right, bottom-left, bottom-right)
                *indices++ = (y + 0) * grid_width + (x + 1);
                *indices++ = (y + 1) * grid_width + (x + 0);
                *indices++ = (y + 1) * grid_width + (x + 1);
            }
        }
        this->ctx.copy_buffer(this->index_buffer, staging_buffer, index_buffer_size);

        // Clean up the staging resources
        this->ctx.vkd->unmap_memory(this->ctx.dev, staging_memory);
        this->ctx.vkd->free_memory(this->ctx.dev, staging_memory, nullptr);
        this->ctx.vkd->destroy_buffer(this->ctx.dev, staging_buffer, nullptr);
    }

    /// \brief Create texture resources including shared image, memory, view, sampler, and descriptor resources.
    void create_texture_resources()
    {
        // 1. Create Shared Image (RGBA32F format, externally sharable)
        VkImageCreateInfo image_create_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_create_info.imageType         = VK_IMAGE_TYPE_2D;
        image_create_info.format
            = VK_FORMAT_R32G32B32A32_SFLOAT; // RGBA32F format for high precision
        image_create_info.extent      = {grid_width, grid_height, 1};
        image_create_info.mipLevels   = 1;
        image_create_info.arrayLayers = 1;
        image_create_info.samples     = VK_SAMPLE_COUNT_1_BIT;
        image_create_info.tiling      = VK_IMAGE_TILING_OPTIMAL;
        image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image_create_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // Setup export capability for the image
        VkExternalMemoryImageCreateInfo ext_image_create_info
            = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
#ifdef _WIN64
        ext_image_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
        ext_image_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        image_create_info.pNext = &ext_image_create_info;
        VK_CHECK(ctx.vkd->create_image(ctx.dev, &image_create_info, nullptr, &height_image));

        // 2. Allocate Image Memory with external and dedicated flags
        VkMemoryRequirements mem_req;
        ctx.vkd->get_image_memory_requirements(ctx.dev, height_image, &mem_req);
        VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize       = mem_req.size;
        alloc_info.memoryTypeIndex
            = ctx.find_memory_type_index(mem_req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Setup export capability for the memory
        VkExportMemoryAllocateInfo export_alloc_info
            = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
#ifdef _WIN64
        export_alloc_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
        export_alloc_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

        // Setup dedicated allocation for the image memory
        VkMemoryDedicatedAllocateInfo dedicated_alloc_info
            = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated_alloc_info.image = height_image;
        export_alloc_info.pNext    = &dedicated_alloc_info;
        alloc_info.pNext           = &export_alloc_info;

        VK_CHECK(ctx.vkd->allocate_memory(ctx.dev, &alloc_info, nullptr, &height_image_memory));
        VK_CHECK(ctx.vkd->bind_image_memory(ctx.dev, height_image, height_image_memory, 0));

        // 3. Create Image View for shader access
        VkImageViewCreateInfo view_create_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_create_info.image                 = height_image;
        view_create_info.viewType              = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format                = image_create_info.format;
        view_create_info.subresourceRange      = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(
            ctx.vkd->create_image_view(ctx.dev, &view_create_info, nullptr, &height_image_view));

        // 4. Create Sampler for texture sampling in the fragment shader
        VkSamplerCreateInfo sampler_create_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_create_info.magFilter           = VK_FILTER_LINEAR;
        sampler_create_info.minFilter           = VK_FILTER_LINEAR;
        sampler_create_info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create_info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create_info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create_info.maxLod              = 0.0f; // No mipmapping needed
        VK_CHECK(ctx.vkd->create_sampler(ctx.dev, &sampler_create_info, nullptr, &texture_sampler));

        // 5. Create Descriptor Set Layout for texture sampler
        VkDescriptorSetLayoutBinding sampler_layout_binding = {};
        sampler_layout_binding.binding                      = 0;
        sampler_layout_binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_layout_binding.descriptorCount = 1;
        sampler_layout_binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info
            = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 1;
        layout_info.pBindings    = &sampler_layout_binding;
        VK_CHECK(ctx.vkd->create_descriptor_set_layout(ctx.dev,
                                                       &layout_info,
                                                       nullptr,
                                                       &descriptor_set_layout));

        // 6. Create Descriptor Pool
        VkDescriptorPoolSize       pool_size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.poolSizeCount              = 1;
        pool_info.pPoolSizes                 = &pool_size;
        pool_info.maxSets                    = 1;
        VK_CHECK(ctx.vkd->create_descriptor_pool(ctx.dev, &pool_info, nullptr, &descriptor_pool));

        // 7. Allocate Descriptor Set
        VkDescriptorSetAllocateInfo alloc_set_info
            = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_set_info.descriptorPool     = descriptor_pool;
        alloc_set_info.descriptorSetCount = 1;
        alloc_set_info.pSetLayouts        = &descriptor_set_layout;
        VK_CHECK(ctx.vkd->allocate_descriptor_sets(ctx.dev, &alloc_set_info, &descriptor_set));

        // 8. Update Descriptor Set with image and sampler info
        VkDescriptorImageInfo image_info = {};
        image_info.imageLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView             = height_image_view;
        image_info.sampler               = texture_sampler;

        VkWriteDescriptorSet descriptor_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        descriptor_write.dstSet               = descriptor_set;
        descriptor_write.dstBinding           = 0;
        descriptor_write.descriptorType       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_write.descriptorCount      = 1;
        descriptor_write.pImageInfo           = &image_info;
        ctx.vkd->update_descriptor_sets(ctx.dev, 1, &descriptor_write, 0, nullptr);

        // 9. Initial Image Layout Transition (Undefined -> General)
        ctx.one_time_submit(
            [&](VkCommandBuffer cmd_buf)
            {
                VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
                barrier.image                = height_image;
                barrier.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                barrier.srcAccessMask        = 0;
                barrier.dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT; // Ready for HIP write

                ctx.vkd->cmd_pipeline_barrier(cmd_buf,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                              0,
                                              0,
                                              nullptr,
                                              0,
                                              nullptr,
                                              1,
                                              &barrier);
            });
    }

    /// \brief Initialize the HIP resources for interoperation with the Vulkan image.
    void initialize_hip_interop_resources()
    {
        // Get memory requirements for the image
        VkMemoryRequirements mem_req;
        ctx.vkd->get_image_memory_requirements(ctx.dev, height_image, &mem_req);

        // Import Vulkan memory into HIP
        this->hip_image_memory_handle = memory_to_hip(ctx, height_image_memory, mem_req.size);

        // Configure mipmapped array descriptor for RGBA32F format
        hipExternalMemoryMipmappedArrayDesc mip_desc = {};
        mip_desc.offset                              = 0;
        mip_desc.formatDesc = {32, 32, 32, 32, hipChannelFormatKindFloat}; // RGBA32F
        mip_desc.extent     = {grid_width, grid_height, 0}; // Depth 0 for 2D
        mip_desc.flags      = hipArraySurfaceLoadStore; // Enable surface operations
        mip_desc.numLevels  = 1; // No mipmaps

        // Map the external memory to a HIP mipmapped array
        HIP_CHECK(hipExternalMemoryGetMappedMipmappedArray(&this->hip_height_mipmapped_array,
                                                           this->hip_image_memory_handle,
                                                           &mip_desc));

        // Get the level 0 array from the mipmapped array
        hipArray_t level0 = nullptr;
        HIP_CHECK(hipMipmappedArrayGetLevel(&level0, this->hip_height_mipmapped_array, 0));

        // Create a surface object from the array for HIP kernel access
        hipResourceDesc resDesc = {};
        resDesc.resType         = hipResourceTypeArray;
        resDesc.res.array.array = level0;
        HIP_CHECK(hipCreateSurfaceObject(&this->hip_height_surface, &resDesc));
    }

    /// \brief Create semaphores for synchronizing between Vulkan and HIP.
    void create_interop_semaphores()
    {
#if USE_EXTERNAL_SEMAPHORES == 1
        // Create a semaphore to signal when the image is ready for HIP processing
        this->image_ready_for_hip     = create_semaphore(this->ctx, true);
        this->hip_image_ready_for_hip = semaphore_to_hip(this->ctx, this->image_ready_for_hip);

        // Create a semaphore to signal when HIP processing is finished
        this->simulation_finished_for_vk = create_semaphore(this->ctx, true);
        this->hip_simulation_finished_for_vk
            = semaphore_to_hip(this->ctx, this->simulation_finished_for_vk);
#endif
    }

    /// \brief Create the pipeline layout with descriptor set layout.
    void create_pipeline_layout()
    {
        VkPipelineLayoutCreateInfo pipeline_layout_info
            = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts    = &descriptor_set_layout;
        VK_CHECK(ctx.vkd->create_pipeline_layout(ctx.dev,
                                                 &pipeline_layout_info,
                                                 nullptr,
                                                 &pipeline_layout));
    }

    /// \brief Create the Vulkan graphics pipeline for rendering the textured quad.
    void create_pipeline()
    {
        // Create shader modules from SPIR-V bytecode
        VkShaderModule vert
            = create_shader_module(this->ctx, std::size(texture_vert), texture_vert);
        VkShaderModule frag
            = create_shader_module(this->ctx, std::size(texture_frag), texture_frag);

        // Setup shader stages
        VkPipelineShaderStageCreateInfo pssci[2] = {};
        pssci[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    vert,
                    "main"};
        pssci[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    frag,
                    "main"};

        // Setup vertex input: only position coordinates
        VkVertexInputBindingDescription bindings[1] = {
            {0, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX}
        };
        VkVertexInputAttributeDescription attribs[1] = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0}
        };

        // Create the pipeline
        this->pipeline = this->ctx.create_simple_pipeline(this->pipeline_layout,
                                                          this->render_pass,
                                                          pssci,
                                                          2,
                                                          bindings,
                                                          1,
                                                          attribs,
                                                          1);

        // Shader modules are no longer needed after pipeline creation
        this->ctx.vkd->destroy_shader_module(this->ctx.dev, vert, nullptr);
        this->ctx.vkd->destroy_shader_module(this->ctx.dev, frag, nullptr);
    }

    /// \brief Recreate the swapchain and related resources when needed.
    /// \param window - The GLFW window handle.
    /// \return True if swapchain was successfully recreated, false if window is minimized.
    bool recreate_swapchain(GLFWwindow* const window)
    {
        VK_CHECK(this->ctx.vkd->device_wait_idle(this->ctx.dev));
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if(width == 0 || height == 0)
            return false;

        // Destroy old framebuffers before recreating swapchain views
        for(const VkFramebuffer fb : this->framebuffers)
        {
            if(fb)
                this->ctx.vkd->destroy_framebuffer(this->ctx.dev, fb, nullptr);
        }
        this->framebuffers.clear();

        // Recreate the swapchain and its framebuffers
        this->sc.recreate({static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
        this->sc.recreate_framebuffers(this->render_pass, this->framebuffers);
        return true;
    }

    /// \brief Begin rendering a new frame.
    /// \param window - The GLFW window handle.
    /// \return True if frame was successfully started, false otherwise.
    bool begin_frame(GLFWwindow* const window)
    {
        const frame& current_frame = frames[this->frame_index % this->frames.size()];
        current_frame.wait();

        // Acquire the next swapchain image
        const swapchain::present_state present_state
            = this->sc.acquire_next_image(current_frame.image_acquired, frame_timeout);

        // Handle swapchain out-of-date conditions
        if(present_state == swapchain::present_state::out_of_date || this->swapchain_out_of_date)
        {
            if(!this->recreate_swapchain(window))
                return false;
            this->swapchain_out_of_date = false;
            return false; // Skip this frame after recreation
        }
        else if(present_state == swapchain::present_state::suboptimal)
        {
            this->swapchain_out_of_date = true;
        }

        // Reset frame resources and begin command buffer
        current_frame.reset();
        VK_CHECK(this->ctx.vkd->reset_command_pool(this->ctx.dev, current_frame.cmd_pool, 0));
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                               nullptr,
                                               VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_CHECK(this->ctx.vkd->begin_command_buffer(current_frame.cmd_buf, &begin_info));
        return true;
    }

    /// \brief Finish rendering the current frame and submit it for presentation.
    void end_frame()
    {
        const frame& current_frame = frames[this->frame_index % this->frames.size()];
        VK_CHECK(this->ctx.vkd->end_command_buffer(current_frame.cmd_buf));

        // Setup semaphores to wait on before rendering
        std::vector<VkSemaphore>          wait_semaphores;
        std::vector<VkPipelineStageFlags> wait_stages;
        wait_semaphores.push_back(current_frame.image_acquired);
        wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        // If HIP-Vulkan sync is enabled, wait for HIP to finish simulation
#if USE_EXTERNAL_SEMAPHORES == 1 && USE_SIGNAL_SEMAPHORE == 1
        wait_semaphores.push_back(this->simulation_finished_for_vk);
        wait_stages.push_back(
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT); // Wait before sampling texture
#endif

        // Setup semaphores to signal after rendering
        std::vector<VkSemaphore> signal_semaphores;
        signal_semaphores.push_back(current_frame.render_finished);
#if USE_EXTERNAL_SEMAPHORES == 1
        signal_semaphores.push_back(this->image_ready_for_hip);
#endif

        // Submit command buffer to graphics queue
        VkSubmitInfo submit_info         = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.waitSemaphoreCount   = static_cast<uint32_t>(wait_semaphores.size());
        submit_info.pWaitSemaphores      = wait_semaphores.data();
        submit_info.pWaitDstStageMask    = wait_stages.data();
        submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
        submit_info.pSignalSemaphores    = signal_semaphores.data();
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &current_frame.cmd_buf;

        VK_CHECK(this->ctx.vkd->queue_submit(this->ctx.graphics_queue.queue,
                                             1,
                                             &submit_info,
                                             current_frame.frame_fence));

        // Present the rendered image
        const swapchain::present_state present_state
            = this->sc.present(current_frame.render_finished);
        if(present_state != swapchain::present_state::optimal)
        {
            this->swapchain_out_of_date = true;
        }
        ++this->frame_index;
    }

    /// \brief Run the HIP tunnel effect simulation for the current frame.
    void step_simulation()
    {
        // Wait for the image to be ready for HIP processing if using external semaphores
#if USE_EXTERNAL_SEMAPHORES == 1
        if(this->frame_index != 0) // Skip first frame as semaphore hasn't been signaled yet
        {
            hipExternalSemaphoreWaitParams wait_params = {};
            HIP_CHECK(hipWaitExternalSemaphoresAsync(&this->hip_image_ready_for_hip,
                                                     &wait_params,
                                                     1,
                                                     this->hip_stream));
        }
#else
        // If not using semaphores, do a full queue sync
        VK_CHECK(this->ctx.vkd->queue_wait_idle(this->ctx.graphics_queue.queue));
#endif

        // Calculate current animation time
        const auto  now  = std::chrono::high_resolution_clock::now();
        const float time = std::chrono::duration<float>(now - this->start_time).count();

        // Configure tunnel animation parameters
        const float speed        = 1.5f; // Speed of forward motion
        const float pattern_freq = 0.5f; // Density of patterns
        const float twist        = 0.1f; // Amount of spiral twist

        // Launch the tunnel kernel with appropriate thread block configuration
        constexpr size_t tile_size = 16; // Thread block size (can be tuned for performance)
        tunnel_kernel<<<dim3(ceiling_div(grid_width, tile_size),
                             ceiling_div(grid_height, tile_size)),
                        dim3(tile_size, tile_size),
                        0,
                        this->hip_stream>>>(this->hip_height_surface,
                                            grid_width,
                                            grid_height,
                                            time,
                                            speed,
                                            pattern_freq,
                                            twist);
        HIP_CHECK(hipGetLastError()); // Check for kernel launch errors

        // Signal that HIP processing is complete
#if USE_EXTERNAL_SEMAPHORES == 1 && USE_SIGNAL_SEMAPHORE == 1
        hipExternalSemaphoreSignalParams signal_params = {};
        HIP_CHECK(hipSignalExternalSemaphoresAsync(&this->hip_simulation_finished_for_vk,
                                                   &signal_params,
                                                   1,
                                                   this->hip_stream));
#else
        // If not using semaphores, synchronize the HIP stream
        HIP_CHECK(hipStreamSynchronize(this->hip_stream));
#endif
    }

    /// \brief Record the Vulkan commands for rendering the current frame.
    /// \param cmd_buf - The command buffer to record commands into.
    void record_draw_commands(VkCommandBuffer cmd_buf)
    {
        const device_dispatch& vkd = *this->ctx.vkd;

        // Transition shared image from General to Shader Read Only layout
        VkImageMemoryBarrier barrier_read = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier_read.oldLayout            = VK_IMAGE_LAYOUT_GENERAL;
        barrier_read.newLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier_read.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        barrier_read.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        barrier_read.image                = height_image;
        barrier_read.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier_read.srcAccessMask        = VK_ACCESS_SHADER_WRITE_BIT; // Written by HIP
        barrier_read.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT; // Read by fragment shader

        vkd.cmd_pipeline_barrier(cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier_read);

        // Begin render pass
        VkClearValue clear_color = {
            {0.1f, 0.1f, 0.1f, 1.0f}
        };
        VkRenderPassBeginInfo rp_begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp_begin_info.renderPass            = this->render_pass;
        rp_begin_info.framebuffer           = this->framebuffers[this->sc.image_index];
        rp_begin_info.renderArea.offset     = {0, 0};
        rp_begin_info.renderArea.extent     = this->sc.extent;
        rp_begin_info.clearValueCount       = 1;
        rp_begin_info.pClearValues          = &clear_color;

        vkd.cmd_begin_render_pass(cmd_buf, &rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        // Set viewport and scissor
        VkViewport viewport
            = {0.0f, 0.0f, (float)this->sc.extent.width, (float)this->sc.extent.height, 0.0f, 1.0f};
        VkRect2D scissor = {
            {0, 0},
            this->sc.extent
        };

        vkd.cmd_set_viewport(cmd_buf, 0, 1, &viewport);
        vkd.cmd_set_scissor(cmd_buf, 0, 1, &scissor);

        // Bind pipeline, descriptor set, vertex and index buffers
        vkd.cmd_bind_pipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipeline);

        VkBuffer     vertex_buffers[] = {this->grid_buffer};
        VkDeviceSize offsets[]        = {0};

        vkd.cmd_bind_vertex_buffers(cmd_buf, 0, 1, vertex_buffers, offsets);
        vkd.cmd_bind_index_buffer(cmd_buf, this->index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkd.cmd_bind_descriptor_sets(cmd_buf,
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline_layout,
                                     0,
                                     1,
                                     &descriptor_set,
                                     0,
                                     nullptr);

        // Draw the triangle mesh
        vkd.cmd_draw_indexed(cmd_buf, num_indices, 1, 0, 0, 0);

        // End render pass
        vkd.cmd_end_render_pass(cmd_buf);

        // Transition image back to General layout for HIP write access
        VkImageMemoryBarrier barrier_write = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier_write.oldLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier_write.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
        barrier_write.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        barrier_write.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        barrier_write.image                = height_image;
        barrier_write.subresourceRange     = barrier_read.subresourceRange;
        barrier_write.srcAccessMask        = VK_ACCESS_SHADER_READ_BIT; // Read by Vulkan
        barrier_write.dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT; // Will be written by HIP

        vkd.cmd_pipeline_barrier(cmd_buf,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier_write);
    }

    /// \brief Render a complete frame, including HIP simulation and Vulkan rendering.
    /// \param window - The GLFW window handle.
    void draw(GLFWwindow* const window)
    {
        // Skip this frame if we can't begin rendering
        if(!this->begin_frame(window))
            return;

        // Run the HIP tunnel simulation
        this->step_simulation();

        // Record and submit Vulkan rendering commands
        const frame& current_frame = frames[this->frame_index % this->frames.size()];
        record_draw_commands(current_frame.cmd_buf);
        this->end_frame();

        // Calculate and display FPS periodically
        const auto frame_time = std::chrono::high_resolution_clock::now();
        const auto time_diff  = frame_time - this->fps_start_time;
        if(time_diff > std::chrono::seconds{5})
        {
            const auto     time_diff_sec   = std::chrono::duration<float>(time_diff).count();
            const uint32_t frames_rendered = this->frame_index - this->fps_start_frame;
            if(frames_rendered > 0 && time_diff_sec > 0)
            {
                std::cout << "Avg FPS: "
                          << double_precision(frames_rendered / time_diff_sec, 2, true) << " ("
                          << double_precision((time_diff_sec * 1000) / frames_rendered, 2, true)
                          << " ms/frame)\n";
            }
            this->fps_start_frame = this->frame_index;
            this->fps_start_time  = frame_time;
        }
    }
};

/// \brief GLFW window resize callback to handle swapchain recreation.
/// \param window - The GLFW window that was resized.
/// \param width - New window width (unused).
/// \param height - New window height (unused).
void resize_callback(GLFWwindow* const window, const int /*width*/, const int /*height*/)
{
    renderer* r = reinterpret_cast<renderer*>(glfwGetWindowUserPointer(window));
    if(r)
        r->swapchain_out_of_date = true;
}

/// \brief Program entry point.
int main()
{
    // The initial size of the GLFW window when the example is first started.
    constexpr VkExtent2D initial_window_extent = {1280, 800};

    // Initialize GLFW with error callback
    glfwSetErrorCallback([](int code, const char* const message)
                         { std::cerr << "GLFW Error: " << message << " (" << code << ")\n"; });
    if(glfwInit() != GLFW_TRUE)
    {
        std::cerr << "Failed to init GLFW\n";
        return error_exit_code;
    }

    // Setup application info and create window
    VkApplicationInfo app_info  = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName   = "HIP-Vulkan interop mipmap example";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_API_VERSION_1_1;
    GLFWwindow* window          = create_window(app_info, initial_window_extent);

    try
    {
        // Initialize Vulkan instance, surface, and find compatible device
        const auto         vkb      = std::make_unique<base_dispatch>(glfwGetInstanceProcAddress);
        const VkInstance   instance = create_instance(*vkb,
                                                    app_info,
                                                    required_instance_extensions,
                                                    std::size(required_instance_extensions));
        const auto         vki      = std::make_unique<instance_dispatch>(*vkb, instance);
        const VkSurfaceKHR surface  = create_surface(instance, window);

        // Find a physical device compatible with both Vulkan and HIP
        physical_device_candidate candidate;
        find_physical_device(*vki, instance, surface, candidate);
        const hipDevice_t hip_device = candidate.hip_candidate.device;

        // Output the selected device information
        hipDeviceProp_t hip_props;
        HIP_CHECK(hipGetDeviceProperties(&hip_props, hip_device));
        std::cout << "Using device: " << candidate.props.deviceName << " (HIP dev " << hip_device
                  << ", UUID " << candidate.hip_candidate.device_uuid << ", CC " << hip_props.major
                  << "." << hip_props.minor << ")\n";

        // Create graphics context, swapchain, and renderer
        graphics_context ctx(vki.get(),
                             instance,
                             surface,
                             candidate.pdev,
                             candidate.queues,
                             required_device_extensions,
                             std::size(required_device_extensions));
        swapchain        swapchain(ctx, initial_window_extent);
        renderer         renderer(ctx, swapchain, hip_device);

        // Set up window callbacks
        glfwSetWindowUserPointer(window, reinterpret_cast<void*>(&renderer));
        glfwSetFramebufferSizeCallback(window, resize_callback);

        // Main rendering loop
        while(glfwWindowShouldClose(window) == GLFW_FALSE)
        {
            glfwPollEvents();
            renderer.draw(window);
        }

        // Clean up GLFW callbacks
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
        // Renderer, swapchain, and context destructors run automatically
    }
    catch(const std::exception& e)
    {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return error_exit_code;
    }
    catch(...)
    {
        std::cerr << "Unknown Runtime Error" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return error_exit_code;
    }

    // Clean up GLFW resources
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
