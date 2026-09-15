#include "../headers/state.hpp"
#include "../headers/context.hpp"

#include <stdexcept>
#include <utility>

namespace cthreads::gpu::memory {

GpuState& GpuState::getInstance() {
    static GpuState instance;
    return instance;
}

GpuState::~GpuState() {
    // Static teardown order vs Context is undefined. Shutdown must clear first
    // so Vulkan handles are already destroyed; only drop the map here.
    std::lock_guard<std::mutex> lock(_mutex);
    _entries.clear();
}

bool GpuState::contains(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.find(name) != _entries.end();
}

size_t GpuState::size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.size();
}

std::vector<std::string> GpuState::names() const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<std::string> out;
    out.reserve(_entries.size());
    for (const auto& pair : _entries) {
        out.push_back(pair.first);
    }
    return out;
}

void GpuState::add(const std::string& name, GpuBuffer&& buffer) {
    if (name.empty()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState.add requires a non-empty "
            "name");
    }
    if (buffer.buffer == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE ||
        buffer.size == 0) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState.add requires a non-empty "
            "GpuBuffer");
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (_entries.find(name) != _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState name already registered: " +
            name);
    }

    GpuStateEntry entry{};
    // GpuBuffer has no custom move that clears handles, so copy then null the
    // caller only after the map insert succeeds (otherwise we would leak).
    entry.buffer = buffer;
    entry.in_use = false;
    _entries.emplace(name, std::move(entry));
    buffer = GpuBuffer{};
}

void GpuState::remove(Context& context, const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    if (it->second.in_use) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState cannot remove in-use "
            "buffer: " +
            name);
    }
    destroy_buffer(context, it->second.buffer);
    _entries.erase(it);
}

GpuBuffer& GpuState::get(const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    return it->second.buffer;
}

const GpuBuffer& GpuState::get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    return it->second.buffer;
}

bool GpuState::is_in_use(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    return it->second.in_use;
}

void GpuState::mark_in_use(const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    if (it->second.in_use) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState buffer already in use: " +
            name);
    }
    it->second.in_use = true;
}

void GpuState::release_in_use(const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    if (!it->second.in_use) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState buffer is not in use: " +
            name);
    }
    it->second.in_use = false;
}

void GpuState::upload(
    Context& context,
    const std::string& name,
    const void* data,
    VkDeviceSize size
) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    if (it->second.in_use) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState cannot upload in-use "
            "buffer: " +
            name);
    }
    // upload_buffer validates size / kind; keep the map lock so remove cannot race.
    upload_buffer(context, it->second.buffer, data, size);
}

void GpuState::download(
    Context& context,
    const std::string& name,
    void* data,
    VkDeviceSize size
) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(name);
    if (it == _entries.end()) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState unknown name: " + name);
    }
    if (it->second.in_use) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: GpuState cannot download in-use "
            "buffer: " +
            name);
    }
    download_buffer(context, it->second.buffer, data, size);
}

void GpuState::clear(Context& context) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& pair : _entries) {
        // Process teardown: destroy even if a launch left in_use set.
        destroy_buffer(context, pair.second.buffer);
        pair.second.in_use = false;
    }
    _entries.clear();
}

} // namespace cthreads::gpu::memory
