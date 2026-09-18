#include "../headers/pack.hpp"
#include "../headers/memory.hpp"
#include "../headers/context.hpp"

#include <stdexcept>
#include <string>

namespace cthreads::gpu::pack {
namespace {

[[noreturn]] void invalid_arg(const char* detail) {
    throw std::invalid_argument(
        std::string("cthreads.gpu.GpuInvalidArgument: ") + detail
    );
}

[[noreturn]] void use_after_destroy(const char* detail) {
    throw std::runtime_error(
        std::string("cthreads.gpu.GpuUseAfterDestroy: ") + detail
    );
}

} // namespace

GpuPack create_gpu_pack(
    cthreads::gpu::Context& context,
    size_t scalar_bytes,
    std::vector<ContainerSpec> container_specs
) {
    GpuPack pack;
    pack.container_slots.reserve(container_specs.size());

    if (scalar_bytes > 0) {
        pack.scalar_buffer = memory::create_buffer(
            context,
            scalar_bytes,
            memory::BufferKind::DeviceLocal
        );
    }

    for (const auto& spec : container_specs) {
        if (spec.elem_bytes == 0) {
            invalid_arg("ContainerSpec must have positive elem_bytes");
        }

        ContainerSlot slot;
        slot.spec = spec;
        if (spec.numel > 0) {
            slot.buffer = memory::create_buffer(
                context,
                static_cast<VkDeviceSize>(spec.numel * spec.elem_bytes),
                memory::BufferKind::DeviceLocal
            );
        }
        pack.container_slots.push_back(std::move(slot));
    }
    return pack;
}

void upload_scalars(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    const void* data,
    size_t size
) {
    if (!data) {
        invalid_arg("data is null");
    }
    if (size == 0) {
        invalid_arg("size must be non-zero");
    }
    if (pack.scalar_buffer.buffer == VK_NULL_HANDLE) {
        use_after_destroy("scalar buffer is not initialized");
    }
    if (size > pack.scalar_buffer.size) {
        invalid_arg("size is greater than the scalar buffer size");
    }

    memory::upload_buffer(context, pack.scalar_buffer, data, size);
}

void upload_container(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    size_t index,
    const void* data,
    size_t size
) {
    if (index >= pack.container_slots.size()) {
        invalid_arg("container index is out of range");
    }
    if (!data) {
        invalid_arg("data is null");
    }
    if (size == 0) {
        invalid_arg("size must be non-zero");
    }

    ContainerSlot& slot = pack.container_slots[index];
    if (slot.spec.numel == 0 || slot.buffer.buffer == VK_NULL_HANDLE) {
        use_after_destroy("container buffer is not initialized");
    }
    const size_t expected = slot.spec.elem_bytes * slot.spec.numel;
    if (size != expected) {
        invalid_arg("size does not match the container size");
    }

    memory::upload_buffer(context, slot.buffer, data, size);
}

void upload_containers(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    const std::vector<const void*>& data,
    const std::vector<size_t>& sizes
) {
    if (data.size() != sizes.size()) {
        invalid_arg("data and sizes must have the same length");
    }
    if (data.size() != pack.container_slots.size()) {
        invalid_arg("data length must match container_slots size");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (pack.container_slots[i].spec.numel == 0) {
            continue;
        }
        try {
            upload_container(context, pack, i, data[i], sizes[i]);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string(e.what()) + " [container " + std::to_string(i) + "]"
            );
        }
    }
}

void download_scalars(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    void* data,
    size_t size
) {
    if (!data) {
        invalid_arg("data is null");
    }
    if (size == 0) {
        invalid_arg("size must be non-zero");
    }
    if (pack.scalar_buffer.buffer == VK_NULL_HANDLE) {
        use_after_destroy("scalar buffer is not initialized");
    }
    if (size > pack.scalar_buffer.size) {
        invalid_arg("size is greater than the scalar buffer size");
    }

    memory::download_buffer(context, pack.scalar_buffer, data, size);
}

void download_container(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    size_t index,
    void* data,
    size_t size
) {
    if (index >= pack.container_slots.size()) {
        invalid_arg("container index is out of range");
    }
    if (!data) {
        invalid_arg("data is null");
    }
    if (size == 0) {
        invalid_arg("size must be non-zero");
    }

    ContainerSlot& slot = pack.container_slots[index];
    if (slot.spec.numel == 0 || slot.buffer.buffer == VK_NULL_HANDLE) {
        use_after_destroy("container buffer is not initialized");
    }
    if (size > slot.buffer.size) {
        invalid_arg("size is greater than the container buffer size");
    }

    memory::download_buffer(context, slot.buffer, data, size);
}

void download_containers(
    cthreads::gpu::Context& context,
    GpuPack& pack,
    std::vector<void*>& data,
    const std::vector<size_t>& sizes
) {
    if (data.size() != sizes.size()) {
        invalid_arg("data and sizes must have the same length");
    }
    if (data.size() != pack.container_slots.size()) {
        invalid_arg("data length must match container_slots size");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (pack.container_slots[i].spec.numel == 0) {
            continue;
        }
        try {
            download_container(context, pack, i, data[i], sizes[i]);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string(e.what()) + " [container " + std::to_string(i) + "]"
            );
        }
    }
}

void destroy_gpu_pack(cthreads::gpu::Context& context, GpuPack& pack) {
    if (pack.scalar_buffer.buffer != VK_NULL_HANDLE) {
        memory::destroy_buffer(context, pack.scalar_buffer);
    }
    for (auto& slot : pack.container_slots) {
        if (slot.buffer.buffer != VK_NULL_HANDLE) {
            memory::destroy_buffer(context, slot.buffer);
        }
    }
    pack = GpuPack{};
}

} // namespace cthreads::gpu::pack
