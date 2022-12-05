// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _HIP_BASIC_VULKAN_INTEROP_VULKAN_UTILS_HPP
#define _HIP_BASIC_VULKAN_INTEROP_VULKAN_UTILS_HPP

#include "example_utils.hpp"

#include <vulkan/vulkan.h>

#ifdef _WIN64
    #define NOMINMAX
    #include <windows.h>

    #include <vulkan/vulkan_win32.h>
#endif

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

/// \brief Checks if the provided Vulkan error code is \p VK_SUCCESS. If not, prints an
/// error message to the standard error output and terminates the program with an error code.
#define VK_CHECK(condition)                                                                   \
    {                                                                                         \
        const VkResult error = condition;                                                     \
        if(error != VK_SUCCESS)                                                               \
        {                                                                                     \
            std::cerr << "A vulkan error encountered: " << error << " at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                               \
            std::exit(error_exit_code);                                                       \
        }                                                                                     \
    }

// Older versions of the vulkan headers define this macro incorrectly to 0, which would
// give compile errors.
#undef VK_NULL_HANDLE
#define VK_NULL_HANDLE nullptr

/// \brief This structure contains the basis function pointers of Vulkan.
///
/// There are two main ways to call the functions of the Vulkan API: Either
/// through the static library (libvulkan-1), or by loading the function pointers
/// manually from the library. When interfacing with other libraries that use
/// vulkan, like GLFW, it is important that all Vulkan functions are invoked from the
/// same library version. In practice, GLFW loads the Vulkan library dynamically by
/// default (and would need to be re-compiled to use the static functions). This means
/// that if, for example, the Vulkan SDK is installed (which contains its own version of
/// libvulkan-1), and GLFW finds that, it might load a different Vulkan library than the
/// statically linked version! The correct approach is thus to ask GLFW for the
/// \p vkGetInstanceProcAddr function that it loaded from the dynamic vulkan library,
/// which is <tt>glfwGetInstanceProcAddress</tt>, and use that to load the other
/// Vulkan functions manually.
///
/// \see https://www.glfw.org/docs/latest/vulkan_guide.html#vulkan_loader.
struct base_dispatch
{
    PFN_vkGetInstanceProcAddr                  get_instance_proc_addr;
    PFN_vkEnumerateInstanceExtensionProperties enumerate_instance_extension_properties;
    PFN_vkCreateInstance                       create_instance;

    /// \brief Initialize a \p base_dispatch by fetching all required base functions from
    /// Vulkan.
    ///
    /// \param loader - The \p vkGetInstanceProcAddr function to load the other function
    ///    pointers with. This can for example be \p glfwGetInstanceProcAddress.
    base_dispatch(PFN_vkGetInstanceProcAddr loader);
};

/// \brief This structure contains the function pointers related to the Vulkan instance.
/// \see base_dispatch
struct instance_dispatch
{
    PFN_vkDestroyInstance                         destroy_instance;
    PFN_vkDestroySurfaceKHR                       destroy_surface;
    PFN_vkEnumeratePhysicalDevices                enumerate_physical_devices;
    PFN_vkGetPhysicalDeviceProperties2            get_physical_device_properties2;
    PFN_vkGetPhysicalDeviceMemoryProperties       get_physical_device_memory_properties;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR      get_physical_device_surface_formats;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR get_physical_device_surface_present_modes;
    PFN_vkEnumerateDeviceExtensionProperties      enumerate_device_extension_properties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties  get_physical_device_queue_family_properties;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR      get_physical_device_surface_support;
    PFN_vkCreateDevice                            create_device;
    PFN_vkGetDeviceProcAddr                       get_device_proc_addr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR get_physical_device_surface_capabilities;

    /// \brief Initialize a \p instance_dispatch by fetching all required base functions
    /// from Vulkan.
    ///
    /// \param instance - The instance to load function pointers for.
    instance_dispatch(const base_dispatch& dispatch, VkInstance instance);
};

/// \brief This structure contains the function pointers for device-specific Vulkan function
/// pointers, for a particular device.
/// \see device_dispatch
struct device_dispatch
{
    PFN_vkDestroyDevice               destroy_device;
    PFN_vkGetDeviceQueue              get_device_queue;
    PFN_vkCreateSwapchainKHR          create_swapchain;
    PFN_vkDestroySwapchainKHR         destroy_swapchain;
    PFN_vkGetSwapchainImagesKHR       get_swapchain_images;
    PFN_vkCreateImageView             create_image_view;
    PFN_vkDestroyImageView            destroy_image_view;
    PFN_vkCreateSemaphore             create_semaphore;
    PFN_vkDestroySemaphore            destroy_semaphore;
    PFN_vkCreateFence                 create_fence;
    PFN_vkDestroyFence                destroy_fence;
    PFN_vkCreateCommandPool           create_command_pool;
    PFN_vkDestroyCommandPool          destroy_command_pool;
    PFN_vkAllocateCommandBuffers      allocate_command_buffers;
    PFN_vkWaitForFences               wait_for_fences;
    PFN_vkResetFences                 reset_fences;
    PFN_vkAcquireNextImageKHR         acquire_next_image;
    PFN_vkQueuePresentKHR             queue_present;
    PFN_vkResetCommandPool            reset_command_pool;
    PFN_vkBeginCommandBuffer          begin_command_buffer;
    PFN_vkEndCommandBuffer            end_command_buffer;
    PFN_vkQueueSubmit                 queue_submit;
    PFN_vkCreateRenderPass            create_render_pass;
    PFN_vkDestroyRenderPass           destroy_render_pass;
    PFN_vkCreateFramebuffer           create_framebuffer;
    PFN_vkDestroyFramebuffer          destroy_framebuffer;
    PFN_vkCreateShaderModule          create_shader_module;
    PFN_vkDestroyShaderModule         destroy_shader_module;
    PFN_vkCreateGraphicsPipelines     create_graphics_pipelines;
    PFN_vkDestroyPipeline             destroy_pipeline;
    PFN_vkCreatePipelineLayout        create_pipeline_layout;
    PFN_vkDestroyPipelineLayout       destroy_pipeline_layout;
    PFN_vkQueueWaitIdle               queue_wait_idle;
    PFN_vkCmdSetViewport              cmd_set_viewport;
    PFN_vkCmdSetScissor               cmd_set_scissor;
    PFN_vkCmdBeginRenderPass          cmd_begin_render_pass;
    PFN_vkCmdBindPipeline             cmd_bind_pipeline;
    PFN_vkCmdEndRenderPass            cmd_end_render_pass;
    PFN_vkCmdDrawIndexed              cmd_draw_indexed;
    PFN_vkCreateBuffer                create_buffer;
    PFN_vkDestroyBuffer               destroy_buffer;
    PFN_vkAllocateMemory              allocate_memory;
    PFN_vkFreeMemory                  free_memory;
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements;
    PFN_vkBindBufferMemory            bind_buffer_memory;
    PFN_vkCmdCopyBuffer               cmd_copy_buffer;
    PFN_vkMapMemory                   map_memory;
    PFN_vkUnmapMemory                 unmap_memory;
    PFN_vkCmdBindVertexBuffers        cmd_bind_vertex_buffers;
    PFN_vkCmdBindIndexBuffer          cmd_bind_index_buffer;
#ifdef _WIN64
    PFN_vkGetMemoryWin32HandleKHR    get_memory_win32_handle;
    PFN_vkGetSemaphoreWin32HandleKHR get_semaphore_win32_handle;
#else
    PFN_vkGetMemoryFdKHR    get_memory_fd;
    PFN_vkGetSemaphoreFdKHR get_semaphore_fd;
#endif

    /// \brief Initialize a \p device_dispatch by fetching all required base functions
    /// from Vulkan.
    ///
    /// \param device - The device to load function pointers for.
    device_dispatch(const instance_dispatch& dispatch, VkDevice device);
};

/// \brief Initialize a GLFW window with a particular extent. The window name is set from
/// the \p app_info.
GLFWwindow* create_window(const VkApplicationInfo& app_info, const VkExtent2D extent);

/// \brief A utility function to check whether all required extensions are supported by Vulkan.
/// Returns \p true if all required extensions are supported.
///
/// \param supported_extensions_properties - The supported extensions as reported by Vulkan, for example
///   by \p vkEnumerateInstanceExtensionProperties or by \p vkEnumerateDeviceExtensionProperties.
/// \param required_extensions_begin - Beginning of the iterator that indicates the extensions we need
///   to be supported. This should be an iterator over <tt>const char*</tt>.
/// \param required_extensions_end - End of the iterator that indicates the extensions that need
///   to be supported.
template<typename IteratorT>
bool extensions_supported(const std::vector<VkExtensionProperties>& supported_extensions_properties,
                          const IteratorT                           required_extensions_begin,
                          const IteratorT                           required_extensions_end)
{
    IteratorT it = required_extensions_begin;
    for(; it != required_extensions_end; ++it)
    {
        const auto supported_it
            = std::find_if(supported_extensions_properties.begin(),
                           supported_extensions_properties.end(),
                           [&](const VkExtensionProperties& props)
                           { return std::strcmp(*it, props.extensionName) == 0; });

        if(supported_it == supported_extensions_properties.end())
            return false;
    }

    return true;
}

/// \brief Create a new Vulkan instance.
/// \param app_info - The application info used to construct the Vulkan instance.
/// \param required_extensions - The extensions to initialize this Vulkan instance with.
///    The required GLFW extensions are added to this value.
/// \param num_required_extensions - The number of extensions in \p required_extensions.
/// \param with_validation - Enable the VK_LAYER_KHRONOS_validation standard validation layer.
VkInstance create_instance(const base_dispatch&     dispatch,
                           const VkApplicationInfo& app_info,
                           const char* const* const required_extensions,
                           const size_t             num_required_extensions,
                           const bool               with_validation = true);

/// \brief Create a Vulkan surface from a GLFW window handle.
VkSurfaceKHR create_surface(const VkInstance instance, GLFWwindow* window);

/// \brief Checks whether the physical device supports a surface at all.
/// Returns \p true if the surface is supported.
bool check_surface_support(const instance_dispatch& dispatch,
                           const VkPhysicalDevice   pdev,
                           const VkSurfaceKHR       surface);

/// \brief Check whether a physical device supports the required extensions.
/// Returns \p true if the extensions are supported.
bool check_device_extensions(const instance_dispatch& dispatch,
                             const VkPhysicalDevice   pdev,
                             const char* const* const required_extensions,
                             const size_t             num_required_extensions);

/// \brief This structure represents an assigment of device queues
/// that will be used to render, present, and perform transfers on.
struct queue_allocation
{
    /// The Vulkan graphics queue family that will be used to render the example.
    uint32_t graphics_family;
    /// The Vulkan properties of the graphics queue.
    VkQueueFamilyProperties graphics_family_properties;

    /// The Vulkan present queue family that will be used to draw the example to
    /// the monitor. May be the same as the \p graphics_family.
    uint32_t present_family;
    /// The Vulkan properties of the present queue.
    VkQueueFamilyProperties present_family_properties;
};

/// \brief Try to allocate device queues for a physical device.
/// This function tries to find a graphics and present queue family index.
/// If there is no such queue available, returns false. Otherwise fills \p qa
/// with details surrounding the queues that should be used.
bool allocate_device_queues(const instance_dispatch& dispatch,
                            const VkPhysicalDevice   pdev,
                            const VkSurfaceKHR       surface,
                            queue_allocation&        qa);

/// \brief This function is used to create a Vulkan logical device from a physical device
/// and a queue allocation;
VkDevice create_device(const instance_dispatch& dispatch,
                       const VkPhysicalDevice   pdev,
                       const queue_allocation&  queues,
                       const char* const* const required_extensions,
                       const size_t             num_required_extensions);

/// \brief A utility structure that groups a Vulkan queue handle and the queue family index
/// that it was created from.
struct queue
{
    VkQueue  queue;
    uint32_t family;
};

/// \brief Create Vulkan device queues from a \p queue_allocation, after the logical device
/// has been created.
void create_device_queues(const device_dispatch&  dispatch,
                          const VkDevice          device,
                          const queue_allocation& queues,
                          queue&                  graphics_queue,
                          queue&                  present_queue);

/// \brief This structure is used to group all basic Vulkan-related stuff together: The Vulkan instance,
/// device, queues, properties, etc. It also provides some utility functions that use those types.
struct graphics_context
{
    const instance_dispatch*         vki;
    std::unique_ptr<device_dispatch> vkd;

    VkInstance                       instance;
    VkSurfaceKHR                     surface;
    VkPhysicalDevice                 pdev;
    VkPhysicalDeviceMemoryProperties mem_props;

    VkDevice dev;
    queue    graphics_queue;
    queue    present_queue;

    VkCommandPool one_time_submit_pool;

    /// \brief Initialize a \p graphics_context. This initializes Vulkan, fetches function pointers, creates
    /// Vulkan logical devices and various related handles.
    graphics_context(const instance_dispatch* vki,
                     const VkInstance         instance,
                     const VkSurfaceKHR       surface,
                     const VkPhysicalDevice   pdev,
                     const queue_allocation&  queues,
                     const char* const* const required_device_extensions,
                     const size_t             num_required_device_extensions);

    ~graphics_context();

    graphics_context(const graphics_context&)            = delete;
    graphics_context& operator=(const graphics_context&) = delete;

    graphics_context(graphics_context&&)            = delete;
    graphics_context& operator=(graphics_context&&) = delete;

    /// \brief Utility function to find a Vulkan surface format suitable for rendering to the GLFW window.
    VkSurfaceFormatKHR find_surface_format() const;

    /// \brief Utility function to find a Vulkan present mode suitable for rendering to the GLFW window.
    VkPresentModeKHR find_present_mode() const;

    /// \brief Utility function that returns \p true if the Vulkan queues used for graphics and present
    /// were created from the same family.
    inline bool graphics_queue_is_present_queue() const
    {
        return this->graphics_queue.family == this->present_queue.family;
    }

    /// \brief Utility function that helps us find a Vulkan memory type that satisfy the required
    /// properties and Vulkan memory type bits. Returns the memory type if any such was found,
    /// or exits the program otherwise.
    uint32_t find_memory_type_index(const uint32_t              memory_type_bits,
                                    const VkMemoryPropertyFlags properties) const;

    /// \brief Initialize a Vulkan pipeline layout.
    VkPipelineLayout create_pipeline_layout() const;

    /// \brief Create a Vulkan pipeline. This is a relatively standard pipeline that renders
    /// a triangle list, with culling enabled, no blending, and dynamic state for viewport and scissor.
    VkPipeline create_simple_pipeline(const VkPipelineLayout                   layout,
                                      const VkRenderPass                       render_pass,
                                      const VkPipelineShaderStageCreateInfo*   shaders,
                                      const unsigned int                       num_shaders,
                                      const VkVertexInputBindingDescription*   bindings,
                                      const unsigned int                       num_bindings,
                                      const VkVertexInputAttributeDescription* attribs,
                                      const unsigned int                       num_attribs) const;

    /// \brief A utility function to quickly submit a single-time command buffer to Vulkan.
    /// This can be used for example to upload buffers to the GPU, or other low-frequency
    /// GPU operations. Operations that happen on a per-frame basis should be handled through
    /// the per-frame command buffer in \p frame.
    /// The command buffer is submitted to a graphics queue, meaning that it is only suitable for
    /// commands that can be submitted to such a queue. Note that a graphics queue in Vulkan is also
    /// always capable as transfer queue, and so the command buffer can be used to perform memory
    /// transfers.
    /// This function blocks until the command buffer has finished executing.
    /// \param f - A callback of type <tt>void(VkCommandBuffer)</tt> that records the commands
    ///   to be submitted. \p vkBeginCommandBuffer and \p vkEndCommandBuffer should not be
    ///   called on this buffer.
    /// \see frame
    template<typename F>
    void one_time_submit(F f) const
    {
        // Reset the command pool and allocate a new buffer that we will use to submit commands to.
        VK_CHECK(this->vkd->reset_command_pool(this->dev, this->one_time_submit_pool, 0));

        VkCommandBufferAllocateInfo cmd_buf_allocate_info = {};
        cmd_buf_allocate_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_buf_allocate_info.commandPool        = this->one_time_submit_pool;
        cmd_buf_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_buf_allocate_info.commandBufferCount = 1;

        VkCommandBuffer cmd_buf;
        VK_CHECK(this->vkd->allocate_command_buffers(this->dev, &cmd_buf_allocate_info, &cmd_buf));

        // Begin recording the command buffer.
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(this->vkd->begin_command_buffer(cmd_buf, &begin_info));

        // Record the commands that we want to dispatch.
        f(cmd_buf);

        // Finalize the command buffer.
        VK_CHECK(this->vkd->end_command_buffer(cmd_buf));

        // Submit the command buffer to the graphics queue.
        VkSubmitInfo submit_info       = {};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf;
        VK_CHECK(
            this->vkd->queue_submit(this->graphics_queue.queue, 1, &submit_info, VK_NULL_HANDLE));

        // Synchronize with the GPU so that we can be sure after this function that the work is finished.
        VK_CHECK(this->vkd->queue_wait_idle(this->graphics_queue.queue));
    }

    /// \brief Utility function to copy two Vulkan buffers.
    void copy_buffer(const VkBuffer dst, const VkBuffer src, const VkDeviceSize size) const;
};

/// \brief Utility function to create a Vulkan shader module from SPIR-V shader byte code.
VkShaderModule create_shader_module(const graphics_context& ctx,
                                    const size_t            shader_len,
                                    const uint32_t*         shader);

/// \brief This structure represents a Vulkan swapchain and all its associated resources. This
/// type is required to let us draw to the GLFW window.
struct swapchain
{
    /// \brief The required usage flags for images owned by the swapchain: We are going to draw to
    /// the swapchain, and so the swapchain images need to be able to act as frame buffer color attachment.
    static constexpr VkImageUsageFlags swapchain_image_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    /// \brief This enumeration represents the current state of the swapchain.
    enum class present_state
    {
        /// Everything is A-OK.
        optimal,
        /// Everything is fine, but the current configuration is not the
        /// optimal. This can for example happen if the window is dragged
        /// to a different monitor that has a different native color format.
        suboptimal,
        /// The swapchain has become outdated with the window's configuration
        /// and should be re-created. This can for example happen if the window
        /// is resized.
        out_of_date,
    };

    /// The graphics context that this swapchain uses.
    const graphics_context& ctx;
    /// The Vulkan handle for the current swapchain.
    VkSwapchainKHR handle;
    /// The format of the surface that we are rendering to.
    VkSurfaceFormatKHR surface_format;
    /// The width and height of the swapchain. Usually corresponds to the dimensions
    /// of the window.
    VkExtent2D extent;
    /// The swapchain's images that we can render to.
    std::vector<VkImage> images;
    /// A Vulkan image view, each index corresponding to the image with the same index in \p images.
    std::vector<VkImageView> views;
    /// The current index of the image that we should draw to. This is an index into
    /// \p images and \p views.
    /// Note: Not valid until acquire_next_image() is called!
    uint32_t image_index;

    /// \brief Create a new Swapchain for a particular surface.
    swapchain(const graphics_context& ctx, VkExtent2D desired_extent);

    ~swapchain();

    swapchain(const swapchain&)            = delete;
    swapchain& operator=(const swapchain&) = delete;

    swapchain(swapchain&&)            = delete;
    swapchain& operator=(swapchain&&) = delete;

    /// \brief The window's extent may not correspond to the actual extent of the swapchain's images.
    /// This function finds the actual resolution that we should render to.
    static VkExtent2D find_actual_extent(const VkSurfaceCapabilitiesKHR& caps,
                                         const VkExtent2D                desired_extent);

    /// \brief Re-create the swapchain after it has become out-of-date or sub-optimal.
    void recreate(VkExtent2D desired_extent);

    /// \brief Fetch the swapchain images that are associated to the current swapchain.
    /// Note: Swapchain images are owned by the swapchain - we do not need to manage their lifetime,
    /// and they should not be destroyed.
    void fetch_swap_images();

    /// \brief Create a Vulkan image view for each swapchain image.
    /// Note: Image views are _not_ owned by the swapchain: We should mind to destroy old views.
    /// This function should be called after <tt>fetch_swap_images</tt>, as it creates a view
    /// for each current image.
    void create_views();

    /// \brief Acquire the next image from the swapchain. This may block until the swapchain has
    /// finished presenting a previous image. This function updates <tt>swapchain::image_index</tt>,
    /// which should be used when rendering.
    /// \returns the current presenting state of the swapchain.
    present_state acquire_next_image(const VkSemaphore image_acquired,
                                     const uint64_t    frame_timeout);

    /// \brief Present the image's contents to the GLFW window.
    /// \param sema - A Vulkan semaphore that the presenting process should wait on before
    ///   continuing with the presenting. This semaphore should be signaled by the Vulkan queue
    ///   submission that renders to the current swapchain image.
    /// \returns the current state of the swapchain.
    present_state present(const VkSemaphore wait_sema) const;

    /// \brief Utility function to create a Vulkan render pass that is compatible
    /// with this swapchain.
    VkRenderPass create_render_pass() const;

    /// \brief (Re-)initialize a vector of framebuffers, each of which is associated to the
    /// swap image with the same index.
    /// \param render_pass - The render pass that each framebuffer is to be associated with.
    /// \param framebuffers - Vector of framebuffers to reinitialize. Elements of this parameter
    ///   will be de-initialized before new elements are inserted.
    void recreate_framebuffers(const VkRenderPass          render_pass,
                               std::vector<VkFramebuffer>& framebuffers);
};

#endif
