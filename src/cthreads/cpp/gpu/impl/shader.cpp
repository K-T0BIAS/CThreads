#include "../headers/shader.hpp"
#include "../headers/shader_cache.hpp"
#include "../headers/context.hpp"

#include <stdexcept>
#include <vector>

namespace cthreads::gpu::shader {

ShaderCacheEntry create_entry(
    Context& context,
    const uint32_t* spirv,
    size_t spirv_word_count,
    uint32_t binding_count
) {
    if (!context.ready || context.device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: create_entry needs an initialized "
            "device");
    }
    if (spirv == nullptr || spirv_word_count == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: create_entry spirv is empty");
    }
    if (binding_count == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: create_entry binding_count must "
            "be >= 1");
    }
    if (!context.vkCreateShaderModule || !context.vkCreateDescriptorSetLayout ||
        !context.vkCreatePipelineLayout || !context.vkCreateComputePipelines) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: create_entry missing shader/"
            "pipeline create entry points");
    }

    ShaderCacheEntry entry{};
    entry.binding_count = binding_count;

    // 1) SPIR-V -> shader module
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv_word_count * sizeof(uint32_t);
    module_info.pCode = spirv;
    if (context.vkCreateShaderModule(
            context.device, &module_info, nullptr, &entry.shader_module) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreateShaderModule failed");
    }

    // 2) set layout: binding i is one STORAGE_BUFFER (compute).
    std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = binding_count;
    layout_info.pBindings = bindings.data();
    if (context.vkCreateDescriptorSetLayout(
            context.device, &layout_info, nullptr, &entry.set_layout) !=
        VK_SUCCESS) {
        destroy_entry(context, entry);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreateDescriptorSetLayout failed");
    }

    // 3) Pipeline layout (one set, no push constants. IF OPTIMIZATION REQUIRES THEM ADD PUSH CONSTS HERE).
    VkPipelineLayoutCreateInfo pipe_layout_info{};
    pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.setLayoutCount = 1;
    pipe_layout_info.pSetLayouts = &entry.set_layout;
    pipe_layout_info.pushConstantRangeCount = 0;
    pipe_layout_info.pPushConstantRanges = nullptr;
    if (context.vkCreatePipelineLayout(
            context.device, &pipe_layout_info, nullptr, &entry.pipeline_layout) !=
        VK_SUCCESS) {
        destroy_entry(context, entry);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreatePipelineLayout failed");
    }

    // 4) Compute pipeline from module + layout (entry point "main").
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = entry.shader_module;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipe_info{};
    pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_info.stage = stage;
    pipe_info.layout = entry.pipeline_layout;
    pipe_info.basePipelineHandle = VK_NULL_HANDLE;
    pipe_info.basePipelineIndex = -1;

    if (context.vkCreateComputePipelines(
            context.device,
            VK_NULL_HANDLE,
            1,
            &pipe_info,
            nullptr,
            &entry.pipeline) != VK_SUCCESS) {
        destroy_entry(context, entry);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: vkCreateComputePipelines failed");
    }

    return entry;
}

} // namespace cthreads::gpu::shader
