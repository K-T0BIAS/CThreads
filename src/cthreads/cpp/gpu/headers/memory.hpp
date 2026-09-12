#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace cthreads::gpu {
struct Context;
}

/**
 * Stateless helpers for one GPU byte region at a time: allocate, free, and
 * copy between host memory and device-local storage buffers.
 *
 * Shader-facing buffers are device-local. Host traffic goes through the
 * Context TransferEngine: reused command pool + fence, and a grow-on-demand
 * host-visible staging scratch. Pass a ready Context each time. GpuPack builds
 * on top.
 *
 * #### Technical terms:
 * - Context: process-wide Vulkan connection (device, queue, loaded entry points).
 * - host: CPU / process memory used by C++ and Python.
 * - device-local: GPU memory optimized for shader access; not persistently mapped.
 * - staging: host-visible buffer used only as a temporary for uploads/downloads.
 * - SSBO: storage buffer - a buffer shaders can read and write.
 */
namespace cthreads::gpu::memory {

/**
 * Which memory path create_buffer should use.
 *
 * One create API, two property sets - not two parallel designs.
 */
enum class BufferKind : uint8_t {
    // Host-visible + coherent; used only as the memcpy side of transfers.
    Staging = 0,
    // Device-local storage buffer for shaders (scalar SSBO or one list SSBO).
    DeviceLocal = 1,
};

/**
 * One contiguous byte region on the GPU (staging scratch, scalar SSBO, or one list).
 *
 * Owns the Vulkan buffer object and the device memory bound to it. GpuPacks
 * use several device-local GpuBuffers: one for all scalars, then one per list.
 *
 * #### Fields:
 * - buffer: VkBuffer = Vulkan handle for the buffer object. VK_NULL_HANDLE if empty.
 * - memory: VkDeviceMemory = allocated memory slab bound to buffer. VK_NULL_HANDLE if empty.
 * - size: VkDeviceSize = caller-facing byte count requested at create (0 if empty).
 * - mapped: void* = CPU pointer when this is a mapped staging buffer; nullptr for
 *   device-local buffers (they are never persistently mapped).
 * - kind: BufferKind = Staging or DeviceLocal; drives destroy and upload/download.
 *
 * #### Technical terms:
 * - VkBuffer: opaque id for a buffer resource (not a raw C pointer to bytes).
 * - VkDeviceMemory: opaque id for an allocated memory block from the driver.
 * - VK_NULL_HANDLE: sentinel meaning no object.
 */
struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    BufferKind kind = BufferKind::DeviceLocal;
};

/**
 * Picks which memory type index on this GPU can back a new buffer.
 *
 * Vulkan gives a bitmask of allowed types for a buffer (type_bits). You also
 * request properties (for example host-visible, or device-local). This walks the
 * device memory types and returns the first index that is allowed and has every
 * requested property flag.
 *
 * #### Parameters:
 * - context: Context& = live GPU context (needs physical device + memory query entry point).
 * - type_bits: uint32_t = bit i set means memory type i is allowed (from memory requirements).
 * - properties: VkMemoryPropertyFlags = required flags for that type.
 *
 * #### Returns:
 * - uint32_t = memory type index for vkAllocateMemory.
 *
 * #### Throws:
 * - runtime_error if context is not ready or no type matches.
 *
 * #### Technical terms:
 * - memory type: one of the heaps/types the driver exposes (device-local, host-visible, etc.).
 * - type_bits: bitmask; bit N means this buffer may use memory type N.
 */
uint32_t find_memory_type(
    cthreads::gpu::Context& context,
    uint32_t type_bits,
    VkMemoryPropertyFlags properties
);

/**
 * Creates a GpuBuffer of the given kind and size.
 *
 * Staging: host-visible + coherent, transfer usage, left mapped for memcpy.
 * DeviceLocal: device-local memory, storage + transfer usage, not mapped.
 * Size must be greater than 0 (empty lists are handled by GpuPack, not a zero-size buffer).
 *
 * #### Parameters:
 * - context: Context& = initialized device and buffer/memory entry points.
 * - size: VkDeviceSize = number of bytes to store (must be > 0).
 * - kind: BufferKind = Staging or DeviceLocal.
 *
 * #### Returns:
 * - GpuBuffer = owned handles; caller must destroy_buffer when done.
 *
 * #### Throws:
 * - runtime_error on zero size, missing entry points, or Vulkan create/allocate/bind/map failure.
 */
GpuBuffer create_buffer(
    cthreads::gpu::Context& context,
    VkDeviceSize size,
    BufferKind kind
);

/**
 * Releases a GpuBuffer and clears its fields.
 *
 * Unmaps if mapped, destroys the VkBuffer, frees VkDeviceMemory, then zeroes the
 * handle. Safe no-op if the buffer is already empty.
 *
 * #### Parameters:
 * - context: Context& = same device that created the buffer.
 * - buffer: GpuBuffer& = allocation to free; left empty after return.
 *
 * #### Returns:
 * - void
 */
void destroy_buffer(cthreads::gpu::Context& context, GpuBuffer& buffer);

/**
 * Copies host bytes into a device-local GpuBuffer (host to device).
 *
 * Final path: memcpy into Context TransferEngine staging, then GPU-copy into
 * the device-local buffer (engine pool + fence).
 *
 * #### Parameters:
 * - context: Context& = device + transfer engine entry points.
 * - buffer: GpuBuffer& = device-local destination; must be large enough.
 * - data: const void* = host source bytes.
 * - size: VkDeviceSize = bytes to copy; must be <= buffer.size.
 *
 * #### Returns:
 * - void
 *
 * #### Throws:
 * - runtime_error if buffer is not device-local, data is null, size is invalid,
 *   transfer engine is missing, or the transfer fails.
 *
 * #### Technical terms:
 * - upload: copy from host (CPU) toward device (GPU) memory.
 */
void upload_buffer(
    cthreads::gpu::Context& context,
    GpuBuffer& buffer,
    const void* data,
    VkDeviceSize size
);

/**
 * Copies bytes from a device-local GpuBuffer into host memory (device to host).
 *
 * Final path: GPU-copy device-local into TransferEngine staging, then memcpy
 * staging to data (engine pool + fence).
 *
 * #### Parameters:
 * - context: Context& = device + transfer engine entry points.
 * - buffer: GpuBuffer& = device-local source; must be large enough.
 * - data: void* = host destination bytes.
 * - size: VkDeviceSize = bytes to copy; must be <= buffer.size.
 *
 * #### Returns:
 * - void
 *
 * #### Throws:
 * - runtime_error if buffer is not device-local, data is null, size is invalid,
 *   transfer engine is missing, or the transfer fails.
 *
 * #### Technical terms:
 * - download: copy from device (GPU) memory to host (CPU).
 * - writeback: copying native results into the same host/Python objects the caller passed in.
 */
void download_buffer(
    cthreads::gpu::Context& context,
    GpuBuffer& buffer,
    void* data,
    VkDeviceSize size
);

} // namespace cthreads::gpu::memory
