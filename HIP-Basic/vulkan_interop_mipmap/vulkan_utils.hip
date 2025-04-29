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

#include "vulkan_utils.hpp"

namespace
{

/// \brief The validation layers that we want to be active by default.
/// The \p VK_LAYER_KHRONOS_validation layer performs general checks on Vulkan
/// calls.
static constexpr const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};

/// \brief A utility function that helps to load a function pointer from Vulkan. If the specified
/// function pointer is not available in the implementation, this function prints an error message
/// and exits the program.
///
/// By passing the target variable as a parameter rather than returning the function
/// pointer, we can perform the cast in this function, and so save a bit of repetitive
/// typing that way.
template<typename FuncType, typename Loader, typename HandleType>
void load_vulkan_function(FuncType& fptr, Loader loader, HandleType handle, const char* const name)
{
    fptr = reinterpret_cast<FuncType>(loader(handle, name));
    if(fptr == nullptr)
    {
        std::cerr << "Failed to load vulkan function pointer " << name << std::endl;
        std::exit(error_exit_code);
    }
}

} // namespace

base_dispatch::base_dispatch(PFN_vkGetInstanceProcAddr loader)
{
    const auto load
        = [&](auto& fptr, const char* name) { load_vulkan_function(fptr, loader, nullptr, name); };

    this->get_instance_proc_addr = loader;
    load(this->enumerate_instance_extension_properties, "vkEnumerateInstanceExtensionProperties");
    load(this->create_instance, "vkCreateInstance");
}

instance_dispatch::instance_dispatch(const base_dispatch& dispatch, VkInstance instance)
{
    const auto load = [&](auto& fptr, const char* name)
    { load_vulkan_function(fptr, dispatch.get_instance_proc_addr, instance, name); };

    load(this->destroy_instance, "vkDestroyInstance");
    load(this->destroy_surface, "vkDestroySurfaceKHR");
    load(this->enumerate_physical_devices, "vkEnumeratePhysicalDevices");
    load(this->get_physical_device_properties2, "vkGetPhysicalDeviceProperties2KHR");
    load(this->get_physical_device_memory_properties, "vkGetPhysicalDeviceMemoryProperties");
    load(this->get_physical_device_surface_formats, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    load(this->get_physical_device_surface_present_modes,
         "vkGetPhysicalDeviceSurfacePresentModesKHR");
    load(this->enumerate_device_extension_properties, "vkEnumerateDeviceExtensionProperties");
    load(this->get_physical_device_queue_family_properties,
         "vkGetPhysicalDeviceQueueFamilyProperties");
    load(this->get_physical_device_surface_support, "vkGetPhysicalDeviceSurfaceSupportKHR");
    load(this->create_device, "vkCreateDevice");
    load(this->get_device_proc_addr, "vkGetDeviceProcAddr");
    load(this->get_physical_device_surface_capabilities,
         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    load(this->get_physical_device_format_properties, "vkGetPhysicalDeviceFormatProperties");
    load(this->get_physical_device_properties, "vkGetPhysicalDeviceProperties");
    load(this->get_physical_device_image_format_properties2,
         "vkGetPhysicalDeviceImageFormatProperties2KHR");
    load(this->get_physical_device_external_semaphore_properties,
         "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR");
}

device_dispatch::device_dispatch(const instance_dispatch& dispatch, VkDevice device)
{
    const auto load = [&](auto& fptr, const char* name)
    { load_vulkan_function(fptr, dispatch.get_device_proc_addr, device, name); };

    load(this->destroy_device, "vkDestroyDevice");
    load(this->get_device_queue, "vkGetDeviceQueue");
    load(this->create_swapchain, "vkCreateSwapchainKHR");
    load(this->destroy_swapchain, "vkDestroySwapchainKHR");
    load(this->get_swapchain_images, "vkGetSwapchainImagesKHR");
    load(this->create_image_view, "vkCreateImageView");
    load(this->destroy_image_view, "vkDestroyImageView");
    load(this->create_semaphore, "vkCreateSemaphore");
    load(this->destroy_semaphore, "vkDestroySemaphore");
    load(this->create_fence, "vkCreateFence");
    load(this->destroy_fence, "vkDestroyFence");
    load(this->create_command_pool, "vkCreateCommandPool");
    load(this->destroy_command_pool, "vkDestroyCommandPool");
    load(this->allocate_command_buffers, "vkAllocateCommandBuffers");
    load(this->wait_for_fences, "vkWaitForFences");
    load(this->reset_fences, "vkResetFences");
    load(this->acquire_next_image, "vkAcquireNextImageKHR");
    load(this->queue_present, "vkQueuePresentKHR");
    load(this->reset_command_pool, "vkResetCommandPool");
    load(this->begin_command_buffer, "vkBeginCommandBuffer");
    load(this->end_command_buffer, "vkEndCommandBuffer");
    load(this->queue_submit, "vkQueueSubmit");
    load(this->create_render_pass, "vkCreateRenderPass");
    load(this->destroy_render_pass, "vkDestroyRenderPass");
    load(this->create_framebuffer, "vkCreateFramebuffer");
    load(this->destroy_framebuffer, "vkDestroyFramebuffer");
    load(this->create_shader_module, "vkCreateShaderModule");
    load(this->destroy_shader_module, "vkDestroyShaderModule");
    load(this->create_graphics_pipelines, "vkCreateGraphicsPipelines");
    load(this->destroy_pipeline, "vkDestroyPipeline");
    load(this->create_pipeline_layout, "vkCreatePipelineLayout");
    load(this->destroy_pipeline_layout, "vkDestroyPipelineLayout");
    load(this->queue_wait_idle, "vkQueueWaitIdle");
    load(this->cmd_set_viewport, "vkCmdSetViewport");
    load(this->cmd_set_scissor, "vkCmdSetScissor");
    load(this->cmd_begin_render_pass, "vkCmdBeginRenderPass");
    load(this->cmd_bind_pipeline, "vkCmdBindPipeline");
    load(this->cmd_end_render_pass, "vkCmdEndRenderPass");
    load(this->cmd_draw_indexed, "vkCmdDrawIndexed");
    load(this->create_buffer, "vkCreateBuffer");
    load(this->destroy_buffer, "vkDestroyBuffer");
    load(this->allocate_memory, "vkAllocateMemory");
    load(this->free_memory, "vkFreeMemory");
    load(this->get_buffer_memory_requirements, "vkGetBufferMemoryRequirements");
    load(this->bind_buffer_memory, "vkBindBufferMemory");
    load(this->cmd_copy_buffer, "vkCmdCopyBuffer");
    load(this->map_memory, "vkMapMemory");
    load(this->unmap_memory, "vkUnmapMemory");
    load(this->cmd_bind_vertex_buffers, "vkCmdBindVertexBuffers");
    load(this->cmd_bind_index_buffer, "vkCmdBindIndexBuffer");
    load(this->create_image, "vkCreateImage");
    load(this->destroy_image, "vkDestroyImage");
    load(this->get_image_memory_requirements, "vkGetImageMemoryRequirements");
    load(this->bind_image_memory, "vkBindImageMemory");
    load(this->cmd_pipeline_barrier, "vkCmdPipelineBarrier");
    load(this->create_descriptor_set_layout, "vkCreateDescriptorSetLayout");
    load(this->destroy_descriptor_set_layout, "vkDestroyDescriptorSetLayout");
    load(this->create_descriptor_pool, "vkCreateDescriptorPool");
    load(this->destroy_descriptor_pool, "vkDestroyDescriptorPool");
    load(this->allocate_descriptor_sets, "vkAllocateDescriptorSets");
    load(this->update_descriptor_sets, "vkUpdateDescriptorSets");
    load(this->create_sampler, "vkCreateSampler");
    load(this->destroy_sampler, "vkDestroySampler");
    load(this->cmd_bind_descriptor_sets, "vkCmdBindDescriptorSets");
    load(this->free_command_buffers, "vkFreeCommandBuffers");
    load(this->device_wait_idle, "vkDeviceWaitIdle");
#ifdef _WIN64
    load(this->get_memory_win32_handle, "vkGetMemoryWin32HandleKHR");
    load(this->get_semaphore_win32_handle, "vkGetSemaphoreWin32HandleKHR");
#else
    load(this->get_memory_fd, "vkGetMemoryFdKHR");
    load(this->get_semaphore_fd, "vkGetSemaphoreFdKHR");
#endif
}

GLFWwindow* create_window(const VkApplicationInfo& app_info, const VkExtent2D extent)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(extent.width,
                                          extent.height,
                                          app_info.pApplicationName,
                                          nullptr,
                                          nullptr);
    if(window == nullptr)
    {
        std::cerr << "Failed to create GLFW window\n";
        std::exit(error_exit_code);
    }

    return window;
}

VkInstance create_instance(const base_dispatch&     dispatch,
                           const VkApplicationInfo& app_info,
                           const char* const* const required_extensions,
                           const size_t             num_required_extensions,
                           const bool               with_validation)
{
    uint32_t                 glfw_extension_count;
    const char* const* const glfw_extensions
        = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> all_required_extensions;
    all_required_extensions.insert(all_required_extensions.end(),
                                   glfw_extensions,
                                   glfw_extensions + glfw_extension_count);
    all_required_extensions.insert(all_required_extensions.end(),
                                   required_extensions,
                                   required_extensions + num_required_extensions);

    uint32_t supported_extension_count;
    VK_CHECK(dispatch.enumerate_instance_extension_properties(nullptr,
                                                              &supported_extension_count,
                                                              nullptr));
    std::vector<VkExtensionProperties> supported_extensions(supported_extension_count);
    VK_CHECK(dispatch.enumerate_instance_extension_properties(nullptr,
                                                              &supported_extension_count,
                                                              supported_extensions.data()));
    if(!extensions_supported(supported_extensions,
                             all_required_extensions.begin(),
                             all_required_extensions.end()))
    {
        std::cerr << "Required instance extensions are not supported\n";
        std::exit(error_exit_code);
    }

    VkInstanceCreateInfo create_info    = {};
    create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo        = &app_info;
    create_info.enabledExtensionCount   = all_required_extensions.size();
    create_info.ppEnabledExtensionNames = all_required_extensions.data();
    if(with_validation)
    {
        create_info.enabledLayerCount   = std::size(validation_layers);
        create_info.ppEnabledLayerNames = validation_layers;
    }

    VkInstance instance;
    VK_CHECK(dispatch.create_instance(&create_info, nullptr, &instance));
    return instance;
}

VkSurfaceKHR create_surface(const VkInstance instance, GLFWwindow* window)
{
    VkSurfaceKHR surface;
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    return surface;
}

bool check_surface_support(const instance_dispatch& dispatch,
                           const VkPhysicalDevice   pdev,
                           const VkSurfaceKHR       surface)
{
    uint32_t format_count;
    VK_CHECK(dispatch.get_physical_device_surface_formats(pdev, surface, &format_count, nullptr));

    uint32_t present_mode_count;
    VK_CHECK(dispatch.get_physical_device_surface_present_modes(pdev,
                                                                surface,
                                                                &present_mode_count,
                                                                nullptr));

    return format_count > 0 && present_mode_count > 0;
}

bool check_device_extensions(const instance_dispatch& dispatch,
                             const VkPhysicalDevice   pdev,
                             const char* const* const required_extensions,
                             const size_t             num_required_extensions)
{
    uint32_t supported_extension_count;
    VK_CHECK(dispatch.enumerate_device_extension_properties(pdev,
                                                            nullptr,
                                                            &supported_extension_count,
                                                            nullptr));
    std::vector<VkExtensionProperties> supported_extensions_properties(supported_extension_count);
    VK_CHECK(
        dispatch.enumerate_device_extension_properties(pdev,
                                                       nullptr,
                                                       &supported_extension_count,
                                                       supported_extensions_properties.data()));

    return extensions_supported(supported_extensions_properties,
                                required_extensions,
                                required_extensions + num_required_extensions);
}

bool allocate_device_queues(const instance_dispatch& dispatch,
                            const VkPhysicalDevice   pdev,
                            const VkSurfaceKHR       surface,
                            queue_allocation&        queues)
{
    uint32_t family_count;
    dispatch.get_physical_device_queue_family_properties(pdev, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    dispatch.get_physical_device_queue_family_properties(pdev, &family_count, families.data());

    int64_t graphics_family = -1;
    int64_t present_family  = -1;
    for(uint32_t i = 0; i < family_count; ++i)
    {
        if(graphics_family < 0 && families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            graphics_family = i;

        VkBool32 present_support = VK_FALSE;
        VK_CHECK(dispatch.get_physical_device_surface_support(pdev, i, surface, &present_support));
        if(present_family < 0 && present_support == VK_TRUE)
            present_family = i;
    }

    if(graphics_family < 0 || present_family < 0)
    {
        return false;
    }

    queues.graphics_family            = graphics_family;
    queues.graphics_family_properties = families[graphics_family];
    queues.present_family             = present_family;
    queues.present_family_properties  = families[present_family];

    return true;
}

VkDevice create_device(const instance_dispatch& dispatch,
                       const VkPhysicalDevice   pdev,
                       const queue_allocation&  queues,
                       const char* const* const required_extensions,
                       const size_t             num_required_extensions)
{
    float                   priorities[] = {1.f, 1.f};
    uint32_t                num_queues;
    VkDeviceQueueCreateInfo queue_create_infos[2] = {};

    // Set up the queue create info for each queue family. We need to consider
    // - The queues may be the same if the queue family supports both graphics and presenting.
    //   In this case we can try to allocate two separate queues if the device has enough queues
    //   for this family. If not, we just use the same queue for both of these operations.
    // - Otherwise we need to allocate to separate queues.
    if(queues.graphics_family == queues.present_family)
    {
        // Queues are the same. Try to allocate 2 if possible, otherwise use the same queue index.
        num_queues                             = 1;
        queue_create_infos[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[0].queueFamilyIndex = queues.graphics_family;
        queue_create_infos[0].queueCount
            = std::min<uint32_t>(2, queues.graphics_family_properties.queueCount);
        queue_create_infos[0].pQueuePriorities = priorities;
    }
    else
    {
        // Different families, so we can allocate them separately.
        num_queues                             = 2;
        queue_create_infos[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[0].queueFamilyIndex = queues.graphics_family;
        queue_create_infos[0].queueCount       = 1;
        queue_create_infos[0].pQueuePriorities = priorities;

        queue_create_infos[1].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[1].queueFamilyIndex = queues.present_family;
        queue_create_infos[1].queueCount       = 1;
        queue_create_infos[1].pQueuePriorities = priorities;
    }

    VkDeviceCreateInfo device_create_info      = {};
    device_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount    = num_queues;
    device_create_info.pQueueCreateInfos       = queue_create_infos;
    device_create_info.ppEnabledExtensionNames = required_extensions;
    device_create_info.enabledExtensionCount   = num_required_extensions;

    VkDevice dev;
    VK_CHECK(dispatch.create_device(pdev, &device_create_info, nullptr, &dev));

    return dev;
}

void create_device_queues(const device_dispatch&  dispatch,
                          const VkDevice          device,
                          const queue_allocation& queues,
                          queue&                  graphics_queue,
                          queue&                  present_queue)
{
    // This function needs to mind the same thing about the device queues:
    // if the candidate and present family are the same, and if the device supports it,
    // we can create two separate queues from the same family.
    uint32_t graphics_index = 0;
    uint32_t present_index  = 0;
    if(queues.graphics_family == queues.present_family
       && queues.graphics_family_properties.queueCount >= 2)
    {
        present_index = 1;
    }

    dispatch.get_device_queue(device,
                              queues.graphics_family,
                              graphics_index,
                              &graphics_queue.queue);
    graphics_queue.family = queues.graphics_family;

    dispatch.get_device_queue(device, queues.present_family, present_index, &present_queue.queue);
    present_queue.family = queues.present_family;
}

graphics_context::graphics_context(const instance_dispatch* vki,
                                   const VkInstance         instance,
                                   const VkSurfaceKHR       surface,
                                   const VkPhysicalDevice   pdev,
                                   const queue_allocation&  queues,
                                   const char* const* const required_device_extensions,
                                   const size_t             num_required_device_extensions)
    : vki(vki), instance(instance), surface(surface), pdev(pdev)
{
    // Fetch some properties of the device which will aid us later.
    this->vki->get_physical_device_memory_properties(this->pdev, &this->mem_props);

    // Create a Vulkan logical device from the physical device candidate, and load the
    // device function pointers.
    this->dev = create_device(*this->vki,
                              this->pdev,
                              queues,
                              required_device_extensions,
                              num_required_device_extensions);
    this->vkd = std::make_unique<device_dispatch>(*this->vki, this->dev);
    create_device_queues(*this->vkd, this->dev, queues, this->graphics_queue, this->present_queue);

    // Create a Vulkan command pool that we can use for one-shot command submissions, like uploading buffers to the
    // device via Vulkan.
    VkCommandPoolCreateInfo cmd_pool_create_info = {};
    cmd_pool_create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_create_info.queueFamilyIndex        = this->graphics_queue.family;
    VK_CHECK(this->vkd->create_command_pool(this->dev,
                                            &cmd_pool_create_info,
                                            nullptr,
                                            &this->one_time_submit_pool));
}

graphics_context::~graphics_context()
{
    this->vkd->destroy_command_pool(this->dev, this->one_time_submit_pool, nullptr);
    this->vkd->destroy_device(this->dev, nullptr);
}

VkSurfaceFormatKHR graphics_context::find_surface_format() const
{
    // TODO: Check that the format has the required format features?
    constexpr VkFormat        preferred_format      = VK_FORMAT_B8G8R8A8_UNORM;
    constexpr VkColorSpaceKHR preferred_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    uint32_t format_count;
    VK_CHECK(this->vki->get_physical_device_surface_formats(this->pdev,
                                                            this->surface,
                                                            &format_count,
                                                            nullptr));
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    VK_CHECK(this->vki->get_physical_device_surface_formats(this->pdev,
                                                            this->surface,
                                                            &format_count,
                                                            formats.data()));

    for(const VkSurfaceFormatKHR format : formats)
    {
        if(format.format == preferred_format && format.colorSpace == preferred_color_space)
        {
            return format;
        }
    }

    return formats[0];
}

VkPresentModeKHR graphics_context::find_present_mode() const
{
    uint32_t present_mode_count;
    VK_CHECK(this->vki->get_physical_device_surface_present_modes(pdev,
                                                                  surface,
                                                                  &present_mode_count,
                                                                  nullptr));
    std::vector<VkPresentModeKHR> modes(present_mode_count);
    VK_CHECK(this->vki->get_physical_device_surface_present_modes(pdev,
                                                                  surface,
                                                                  &present_mode_count,
                                                                  modes.data()));

    constexpr VkPresentModeKHR preferred[]
        = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};

    for(const VkPresentModeKHR mode : modes)
    {
        const auto it = std::find(std::begin(preferred), std::end(preferred), mode);
        if(it != std::end(preferred))
            return mode;
    }

    return VK_PRESENT_MODE_FIFO_KHR; // always supported
}

uint32_t graphics_context::find_memory_type_index(const uint32_t              memory_type_bits,
                                                  const VkMemoryPropertyFlags properties) const
{
    for(uint32_t i = 0; i < this->mem_props.memoryTypeCount; ++i)
    {
        if((memory_type_bits & (1U << i)) == 0)
            continue;
        if((this->mem_props.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    std::cerr << "failed to find a suitable memory type\n";
    std::exit(error_exit_code);
}

VkPipelineLayout graphics_context::create_pipeline_layout() const
{
    VkPipelineLayoutCreateInfo create_info = {};
    create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    VkPipelineLayout pipeline_layout;
    VK_CHECK(this->vkd->create_pipeline_layout(this->dev, &create_info, nullptr, &pipeline_layout));

    return pipeline_layout;
}

VkPipeline
    graphics_context::create_simple_pipeline(const VkPipelineLayout                   layout,
                                             const VkRenderPass                       render_pass,
                                             const VkPipelineShaderStageCreateInfo*   shaders,
                                             const unsigned int                       num_shaders,
                                             const VkVertexInputBindingDescription*   bindings,
                                             const unsigned int                       num_bindings,
                                             const VkVertexInputAttributeDescription* attribs,
                                             const unsigned int num_attribs) const
{
    VkPipelineVertexInputStateCreateInfo pvisci = {};
    pvisci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pvisci.vertexBindingDescriptionCount   = num_bindings;
    pvisci.pVertexBindingDescriptions      = bindings;
    pvisci.vertexAttributeDescriptionCount = num_attribs;
    pvisci.pVertexAttributeDescriptions    = attribs;

    VkPipelineInputAssemblyStateCreateInfo piasci = {};
    piasci.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    piasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo pvsci = {};
    pvsci.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pvsci.viewportCount                     = 1; // set with cmdSetViewport
    pvsci.scissorCount                      = 1; // set with cmdSetScissor

    VkDynamicState dynstate[]              = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo pdsci = {};
    pdsci.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pdsci.dynamicStateCount                = std::size(dynstate);
    pdsci.pDynamicStates                   = dynstate;

    VkPipelineRasterizationStateCreateInfo prsci = {};
    prsci.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    prsci.polygonMode = VK_POLYGON_MODE_FILL;
    prsci.cullMode    = VK_CULL_MODE_BACK_BIT;
    prsci.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    prsci.lineWidth   = 1.f;

    VkPipelineMultisampleStateCreateInfo pmsci = {};
    pmsci.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pmsci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pmsci.minSampleShading     = 1;

    VkPipelineColorBlendAttachmentState pcbas = {};
    pcbas.srcColorBlendFactor                 = VK_BLEND_FACTOR_ONE;
    pcbas.dstColorBlendFactor                 = VK_BLEND_FACTOR_ZERO;
    pcbas.colorBlendOp                        = VK_BLEND_OP_ADD;
    pcbas.srcAlphaBlendFactor                 = VK_BLEND_FACTOR_ONE;
    pcbas.dstAlphaBlendFactor                 = VK_BLEND_FACTOR_ZERO;
    pcbas.alphaBlendOp                        = VK_BLEND_OP_ADD;
    pcbas.colorWriteMask                      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                           | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo pcbsci = {};
    pcbsci.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pcbsci.logicOp         = VK_LOGIC_OP_COPY;
    pcbsci.attachmentCount = 1;
    pcbsci.pAttachments    = &pcbas;

    VkGraphicsPipelineCreateInfo create_info = {};
    create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    create_info.stageCount                   = num_shaders;
    create_info.pStages                      = shaders;
    create_info.pVertexInputState            = &pvisci;
    create_info.pInputAssemblyState          = &piasci;
    create_info.pViewportState               = &pvsci;
    create_info.pRasterizationState          = &prsci;
    create_info.pMultisampleState            = &pmsci;
    create_info.pColorBlendState             = &pcbsci;
    create_info.pDynamicState                = &pdsci;
    create_info.layout                       = layout;
    create_info.renderPass                   = render_pass;
    create_info.subpass                      = 0;

    VkPipeline pipeline;
    VK_CHECK(this->vkd->create_graphics_pipelines(this->dev,
                                                  VK_NULL_HANDLE,
                                                  1,
                                                  &create_info,
                                                  nullptr,
                                                  &pipeline));

    return pipeline;
}

void graphics_context::copy_buffer(const VkBuffer     dst,
                                   const VkBuffer     src,
                                   const VkDeviceSize size) const
{
    this->one_time_submit(
        [&](const VkCommandBuffer cmd_buf)
        {
            VkBufferCopy region = {};
            region.srcOffset    = 0;
            region.dstOffset    = 0;
            region.size         = size;
            this->vkd->cmd_copy_buffer(cmd_buf, src, dst, 1, &region);
        });
}

VkCommandBuffer graphics_context::begin_single_time_commands() const
{
    VkCommandBufferAllocateInfo allocate_info = {};
    allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool                 = this->one_time_submit_pool;
    allocate_info.commandBufferCount          = 1;

    VkCommandBuffer command_buffer;
    VK_CHECK(this->vkd->allocate_command_buffers(this->dev, &allocate_info, &command_buffer));

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(this->vkd->begin_command_buffer(command_buffer, &begin_info));
    return command_buffer;
}

void graphics_context::end_single_time_commands(VkCommandBuffer command_buffer) const
{
    VK_CHECK(this->vkd->end_command_buffer(command_buffer));

    VkSubmitInfo submit_info       = {};
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &command_buffer;

    VK_CHECK(this->vkd->queue_submit(this->graphics_queue.queue, 1, &submit_info, VK_NULL_HANDLE));
    VK_CHECK(this->vkd->queue_wait_idle(this->graphics_queue.queue));

    this->vkd->free_command_buffers(this->dev, this->one_time_submit_pool, 1, &command_buffer);
}

VkShaderModule create_shader_module(const graphics_context& ctx,
                                    const size_t            shader_len,
                                    const uint32_t*         shader)
{
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize                 = sizeof(uint32_t) * shader_len;
    create_info.pCode                    = shader;

    VkShaderModule module;
    VK_CHECK(ctx.vkd->create_shader_module(ctx.dev, &create_info, nullptr, &module));
    return module;
}

swapchain::swapchain(const graphics_context& ctx, VkExtent2D desired_extent)
    : ctx(ctx), handle(VK_NULL_HANDLE)
{
    this->recreate(desired_extent);
}

swapchain::~swapchain()
{
    for(const VkImageView& view : this->views)
    {
        this->ctx.vkd->destroy_image_view(this->ctx.dev, view, nullptr);
    }

    this->ctx.vkd->destroy_swapchain(this->ctx.dev, this->handle, nullptr);
}

VkExtent2D swapchain::find_actual_extent(const VkSurfaceCapabilitiesKHR& caps,
                                         const VkExtent2D                desired_extent)
{
    if(caps.currentExtent.width != 0xFFFF'FFFF)
    {
        return caps.currentExtent;
    }

    return VkExtent2D{
        std::clamp(desired_extent.width, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(desired_extent.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
}

void swapchain::recreate(VkExtent2D desired_extent)
{
    this->surface_format                = this->ctx.find_surface_format();
    const VkPresentModeKHR present_mode = this->ctx.find_present_mode();

    VkSurfaceCapabilitiesKHR surface_caps;
    VK_CHECK(this->ctx.vki->get_physical_device_surface_capabilities(this->ctx.pdev,
                                                                     this->ctx.surface,
                                                                     &surface_caps));

    this->extent = find_actual_extent(surface_caps, desired_extent);

    if((surface_caps.supportedUsageFlags & swapchain_image_usage) != swapchain_image_usage)
    {
        std::cerr << "Surface does not support intended usage flags\n";
        std::exit(error_exit_code);
    }

    uint32_t image_count = surface_caps.minImageCount + 1;
    if(surface_caps.maxImageCount > 0)
    {
        image_count = std::min(image_count, surface_caps.maxImageCount);
    }

    const uint32_t queue_families[]
        = {this->ctx.graphics_queue.family, this->ctx.present_queue.family};

    const VkSwapchainKHR old_swapchain = this->handle;

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface                  = this->ctx.surface;
    create_info.minImageCount            = image_count;
    create_info.imageFormat              = this->surface_format.format;
    create_info.imageColorSpace          = this->surface_format.colorSpace;
    create_info.imageExtent              = this->extent;
    create_info.imageArrayLayers         = 1;
    create_info.imageUsage               = swapchain_image_usage;
    create_info.imageSharingMode         = this->ctx.graphics_queue_is_present_queue()
                                               ? VK_SHARING_MODE_EXCLUSIVE
                                               : VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount    = this->ctx.graphics_queue_is_present_queue() ? 1 : 2;
    create_info.pQueueFamilyIndices      = queue_families;
    create_info.preTransform             = surface_caps.currentTransform;
    create_info.compositeAlpha           = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode              = present_mode;
    create_info.clipped                  = VK_TRUE;
    create_info.oldSwapchain             = old_swapchain;

    VK_CHECK(this->ctx.vkd->create_swapchain(this->ctx.dev, &create_info, nullptr, &this->handle));

    // Note: it may be better to wait a few frames with destroying the swapchain to give it the
    // time to finalize rendering. We are lazy though and destroy it now.
    if(old_swapchain != VK_NULL_HANDLE)
    {
        this->ctx.vkd->destroy_swapchain(this->ctx.dev, old_swapchain, nullptr);
    }

    this->fetch_swap_images();
    this->create_views();
}

void swapchain::fetch_swap_images()
{
    uint32_t count;
    VK_CHECK(this->ctx.vkd->get_swapchain_images(this->ctx.dev, this->handle, &count, nullptr));
    // Note: Old images do not need to be manually destroyed.
    this->images.resize(count);
    VK_CHECK(this->ctx.vkd->get_swapchain_images(this->ctx.dev,
                                                 this->handle,
                                                 &count,
                                                 this->images.data()));
}

void swapchain::create_views()
{
    // If we are recreating the swapchain, then we need to destroy the old image views.
    for(const VkImageView view : this->views)
    {
        this->ctx.vkd->destroy_image_view(this->ctx.dev, view, nullptr);
    }

    this->views.resize(this->images.size());
    for(size_t i = 0; i < this->images.size(); ++i)
    {
        VkImageViewCreateInfo create_info           = {};
        create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image                           = this->images[i];
        create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format                          = this->surface_format.format;
        create_info.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel   = 0;
        create_info.subresourceRange.levelCount     = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount     = 1;

        VK_CHECK(this->ctx.vkd->create_image_view(this->ctx.dev,
                                                  &create_info,
                                                  nullptr,
                                                  &this->views[i]));
    }
}

swapchain::present_state swapchain::acquire_next_image(const VkSemaphore image_acquired,
                                                       const uint64_t    frame_timeout)
{
    const VkResult result = this->ctx.vkd->acquire_next_image(this->ctx.dev,
                                                              this->handle,
                                                              frame_timeout,
                                                              image_acquired,
                                                              VK_NULL_HANDLE,
                                                              &this->image_index);
    switch(result)
    {
        case VK_SUCCESS: return present_state::optimal;
        case VK_SUBOPTIMAL_KHR: return present_state::suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return present_state::out_of_date;
        case VK_TIMEOUT:
        default: VK_CHECK(result); return present_state::out_of_date; // make compiler happy.
    }
}

swapchain::present_state swapchain::present(const VkSemaphore wait_sema) const
{
    VkPresentInfoKHR present_info   = {};
    present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores    = &wait_sema;
    present_info.swapchainCount     = 1;
    present_info.pSwapchains        = &this->handle;
    present_info.pImageIndices      = &this->image_index;

    const VkResult result
        = this->ctx.vkd->queue_present(this->ctx.present_queue.queue, &present_info);
    switch(result)
    {
        case VK_SUCCESS: return present_state::optimal;
        case VK_SUBOPTIMAL_KHR: return present_state::suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return present_state::out_of_date;
        default: VK_CHECK(result); return present_state::out_of_date; // make compiler happy.
    }
}

/// \brief Utility function to create a Vulkan render pass that is compatible
/// with a particular swapchain.
VkRenderPass swapchain::create_render_pass() const
{
    VkAttachmentDescription color_attachment = {};
    color_attachment.format                  = this->surface_format.format;
    color_attachment.samples                 = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout             = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_attachment_ref;

    VkRenderPassCreateInfo create_info = {};
    create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    create_info.attachmentCount        = 1;
    create_info.pAttachments           = &color_attachment;
    create_info.subpassCount           = 1;
    create_info.pSubpasses             = &subpass;

    VkRenderPass render_pass;
    VK_CHECK(ctx.vkd->create_render_pass(ctx.dev, &create_info, nullptr, &render_pass));
    return render_pass;
}

void swapchain::recreate_framebuffers(const VkRenderPass          render_pass,
                                      std::vector<VkFramebuffer>& framebuffers)
{
    // Be sure to delete the old frame buffers if any exist.
    for(const VkFramebuffer fb : framebuffers)
    {
        this->ctx.vkd->destroy_framebuffer(this->ctx.dev, fb, nullptr);
    }

    framebuffers.resize(this->images.size());
    for(uint32_t i = 0; i < this->images.size(); ++i)
    {
        VkFramebufferCreateInfo create_info = {};
        create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        create_info.renderPass              = render_pass;
        create_info.attachmentCount         = 1;
        create_info.pAttachments            = &this->views[i];
        create_info.width                   = this->extent.width;
        create_info.height                  = this->extent.height;
        create_info.layers                  = 1;

        VK_CHECK(this->ctx.vkd->create_framebuffer(this->ctx.dev,
                                                   &create_info,
                                                   nullptr,
                                                   &framebuffers[i]));
    }
}
