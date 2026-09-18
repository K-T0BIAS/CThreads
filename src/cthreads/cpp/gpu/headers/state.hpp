#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.hpp"

namespace cthreads::gpu {
struct Context;
}

/**
 * Process-wide registry of named device-local GpuBuffers.
 *
 * Owns GPU memory outside of a single launch so kernels can reuse the same
 * buffers across dispatches. Names are unique: adding a duplicate name throws.
 * Intended for a later Python binding of this singleton so host code can
 * allocate, upload, launch, download, and free without leaking or double-owning
 * Vulkan handles.
 *
 * This is not a public DeviceBuffer type. Python should see a controlled state
 * / arena API that calls into this registry; VkBuffer stays inside _ext.
 *
 * #### Technical terms:
 * - GpuBuffer: one contiguous device (or staging) byte region (see memory.hpp).
 * - Context: process-wide Vulkan connection (device, queue, loaded entry points).
 * - in_use: true while a launch has checked out this name; blocks remove and a
 *   second mark_in_use until release_in_use.
 * - singleton: one process-wide instance via getInstance(); not copyable.
 */
namespace cthreads::gpu::memory {

/**
 * One named row in GpuState: owned buffer plus launch checkout flag.
 *
 * #### Fields:
 * - buffer: GpuBuffer = owned Vulkan allocation (moved in on add).
 * - in_use: bool = true while a kernel job holds this name for dispatch.
 */
struct GpuStateEntry {
    GpuBuffer buffer{};
    bool in_use = false;
};

/**
 * Singleton map of unique string names to owned GpuBuffers.
 *
 * Thread-safe: every public method locks an internal mutex. Call clear(context)
 * from Context shutdown before destroying the logical device so VkBuffer /
 * VkDeviceMemory handles are freed while the device is still alive.
 *
 * Writers / readers: any native or pybind caller. Duplicate names are rejected
 * on add. Removing or clearing while in_use is rejected on remove; clear on
 * shutdown still destroys (process teardown).
 */
class GpuState {
private:
    std::unordered_map<std::string, GpuStateEntry> _entries;
    mutable std::mutex _mutex;

    GpuState() = default;
    ~GpuState();

public:
    static GpuState& getInstance();

    GpuState(const GpuState&) = delete;
    GpuState& operator=(const GpuState&) = delete;
    GpuState(GpuState&&) = delete;
    GpuState& operator=(GpuState&&) = delete;

    /**
     * Destroy every registered buffer and empty the map.
     *
     * Call from Context shutdown before destroying the logical device.
     *
     * #### Parameters:
     * - context: Context& = device used to destroy buffers
     */
    void clear(cthreads::gpu::Context& context);

    /**
     * True if `name` is already registered.
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Returns:
     * - bool = true when an entry exists for name
     */
    bool contains(const std::string& name) const;

    /**
     * Number of registered buffers.
     *
     * #### Returns:
     * - size_t = entry count
     */
    size_t size() const;

    /**
     * Snapshot of all registered names (order is not meaningful).
     *
     * #### Returns:
     * - vector<string> = copy of keys for introspection / pybind
     */
    std::vector<std::string> names() const;

    /**
     * Take ownership of a buffer under a unique name.
     *
     * Moves `buffer` into the registry. The caller must not use or destroy the
     * moved-from GpuBuffer afterward (handles are null after a successful add).
     *
     * #### Parameters:
     * - name: const string& = unique key (must be non-empty and not already used)
     * - buffer: GpuBuffer&& = owned allocation to store (typically DeviceLocal)
     *
     * #### Throws:
     * - runtime_error = empty name, duplicate name, or empty buffer handles
     */
    void add(const std::string& name, GpuBuffer&& buffer);

    /**
     * Destroy and unregister one buffer by name.
     *
     * #### Parameters:
     * - context: Context& = same device that created the buffer
     * - name: const string& = registry key
     *
     * #### Throws:
     * - runtime_error = unknown name, or entry is in_use (release first)
     */
    void remove(cthreads::gpu::Context& context, const std::string& name);

    /**
     * Mutable reference to the buffer stored under `name`.
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Returns:
     * - GpuBuffer& = owned buffer (valid until remove/clear)
     *
     * #### Throws:
     * - runtime_error = unknown name
     */
    GpuBuffer& get(const std::string& name);

    /**
     * Const reference to the buffer stored under `name`.
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Returns:
     * - const GpuBuffer& = owned buffer (valid until remove/clear)
     *
     * #### Throws:
     * - runtime_error = unknown name
     */
    const GpuBuffer& get(const std::string& name) const;

    /**
     * True if the named entry is checked out for a launch.
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Returns:
     * - bool = entry.in_use
     *
     * #### Throws:
     * - runtime_error = unknown name
     */
    bool is_in_use(const std::string& name) const;

    /**
     * Mark a buffer as checked out so remove and a second mark fail.
     *
     * Call from the launch path before recording work that binds this buffer.
     * Pair with release_in_use after the fence wait (or on launch failure).
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Throws:
     * - runtime_error = unknown name, or already in_use
     */
    void mark_in_use(const std::string& name);

    /**
     * Clear the in_use flag after a launch finishes (or aborts).
     *
     * #### Parameters:
     * - name: const string& = registry key
     *
     * #### Throws:
     * - runtime_error = unknown name, or entry was not in_use
     */
    void release_in_use(const std::string& name);

    /**
     * Upload host bytes into a registered device-local buffer (H2D).
     *
     * #### Parameters:
     * - context: Context& = same device that created the buffer
     * - name: const string& = registry key
     * - data: const void* = host source bytes
     * - size: VkDeviceSize = bytes to copy; must be > 0 and <= buffer.size
     *
     * #### Throws:
     * - runtime_error = unknown name, in_use, bad size, or transfer failure
     */
    void upload(
        cthreads::gpu::Context& context,
        const std::string& name,
        const void* data,
        VkDeviceSize size
    );

    /**
     * Download a registered device-local buffer into host bytes (D2H).
     *
     * #### Parameters:
     * - context: Context& = same device that created the buffer
     * - name: const string& = registry key
     * - data: void* = host destination bytes
     * - size: VkDeviceSize = bytes to copy; must be > 0 and <= buffer.size
     *
     * #### Throws:
     * - runtime_error = unknown name, in_use, bad size, or transfer failure
     */
    void download(
        cthreads::gpu::Context& context,
        const std::string& name,
        void* data,
        VkDeviceSize size
    );
};

} // namespace cthreads::gpu::memory
