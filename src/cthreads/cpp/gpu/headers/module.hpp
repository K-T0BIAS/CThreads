#pragma once

#include <cstdint>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <pybind11/pybind11.h>

#include "pack.hpp"
#include "descriptors.hpp"

namespace py = pybind11;

namespace cthreads::gpu {

struct Context;

/**
 * Per-launch GPU job handle (mirror of CPU SpawnedKernel, without an OS thread).
 *
 * CPU kernels run on a CThread. GPU kernels run on the device after
 * vkQueueSubmit; this struct owns the host-side inflight state and waits on a
 * fence in join(). There is no CGpuThread.
 *
 * Typical lifetime:
 * 1. Launch path fills pack, descriptor set, records/submits with fence.
 * 2. Caller may overlap other host work.
 * 3. join() waits on the fence, downloads, writeback, then releases GPU objects.
 *
 * Methods are declared here; record/submit/join bodies land with the launch path.
 *
 * #### Fields:
 * - pack: GpuPack = device-local scalar + list SSBOs for this launch.
 * - descriptor_pool: DescriptorPool = pool that allocated descriptor_set (for free_set).
 * - descriptor_set: VkDescriptorSet = bindings wired to pack buffers.
 * - command_buffer: VkCommandBuffer = checked out from Context LaunchEngine.
 * - command_pool: VkCommandPool = Context launch pool (borrowed; not destroyed on join).
 * - fence: VkFence = checked out per job; returned to LaunchEngine after wait.
 * - symbol: string = shader cache key for this kernel.
 * - group_count_x/y/z: uint32_t = vkCmdDispatch workgroup counts.
 * - values_keep: shared_ptr to py::list = Python args kept alive for list writeback.
 * - writeback_lists: plan of ref list slots to download into values_keep on join.
 * - finished: bool = true after join completed writeback / teardown.
 * - eptr: exception_ptr = error captured during launch or join (rethrown on join).
 *
 * #### Technical terms:
 * - Fence: GPU timeline object the CPU waits on until submitted work completes.
 * - Descriptor set: per-launch table binding i -> pack buffer i.
 * - GpuPack: binding-convention buffer bag (scalars at 0, lists at 1..N).
 */
struct SpawnedGpuKernel {
    /**
     * One list argument to download into the kept Python list on join.
     * Built at launch from meta (pass_as ref) + live numel.
     */
    struct WritebackListSlot {
        size_t value_index = 0;     // index into values_keep
        size_t container_index = 0; // index into pack.container_slots
        size_t numel = 0;
        std::string elem_kind;      // "float" / "int" / "double" / "bool"

    };

    pack::GpuPack pack{};
    pack::DescriptorPool descriptor_pool{};
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    std::string symbol;
    uint32_t group_count_x = 1;
    uint32_t group_count_y = 1;
    uint32_t group_count_z = 1;

    // Same Python list objects the caller passed (CPU SpawnedKernel values_keep).
    std::shared_ptr<py::list> values_keep;
    // Ref list slots only; value scalars are not written back.
    std::vector<WritebackListSlot> writeback_lists;

    // Parallel to pack.container_slots: true => destroy_buffer on release.
    // False => borrowed from GpuState; handles cleared without destroy.
    std::vector<uint8_t> container_owned;
    // GpuState names marked in_use for this launch; released in release_inflight.
    std::vector<std::string> resident_names;

    bool finished = false;
    std::mutex done_mu;
    std::condition_variable done_cv;
    bool done_flag = false;
    std::exception_ptr eptr;

    SpawnedGpuKernel() = default;
    SpawnedGpuKernel(const SpawnedGpuKernel&) = delete;
    SpawnedGpuKernel& operator=(const SpawnedGpuKernel&) = delete;
    SpawnedGpuKernel(SpawnedGpuKernel&&) = delete;
    SpawnedGpuKernel& operator=(SpawnedGpuKernel&&) = delete;

    /**
     * No-op for the default path (work is submitted at launch time).
     * Kept so Python Job-shaped wrappers can share a start/join/done surface
     * with CPU SpawnedKernel.
     */
    void start();

    /**
     * Wait until the GPU fence signals, optionally download/writeback, then
     * release inflight GPU objects. Rethrows eptr if set. Idempotent after finished.
     *
     * #### Parameters:
     * - context: Context& = same device that created pack / submitted work.
     * - download: bool = if true (default), download ref lists into values_keep.
     *   If false, skip writeback (resident buffers stay device-authoritative).
     */
    void join(Context& context, bool download = true);

    /**
     * Block until done_flag is set (join or failure path). Does not download.
     * Prefer join() for the full writeback and teardown path.
     */
    void wait();

    /**
     * True after the GPU work is complete (done_flag). Does not perform writeback.
     */
    bool done();

    /**
     * Destroy remaining Vulkan objects if join was never called.
     * Safe if already finished / empty.
     */
    ~SpawnedGpuKernel();
};

/**
 * GPU analog of CPU spawn_from_meta: marshal args, submit compute, return a job.
 *
 * Ensures the process Context is ready, resolves the kernel from meta (shader
 * cache symbol / SPIR-V), builds and uploads a GpuPack from ordered_values,
 * allocates and updates a descriptor set, records bind+dispatch, submits with
 * a fence, and returns an owned SpawnedGpuKernel. Does not wait; call
 * job->join(context) for fence wait, download, and writeback.
 *
 * Unlike spawn_from_meta there is no CThread and no pool argument. Work runs on
 * the GPU after vkQueueSubmit on the calling host thread.
 *
 * #### Parameters:
 * - meta: py::dict = kernel metadata (symbol, binding layout / scalar size,
 *   container specs, dispatch size, and later types/schemas for writeback).
 *   Shape will align with @Gpu __kernel_meta__ when emit exists; smoke tests may
 *   pass a minimal dict.
 * - ordered_values: py::list = Python args in parameter order matching meta
 *   (scalars flattened into the scalar SSBO; lists map to bindings 1..N).
 *
 * #### Returns:
 * - shared_ptr<SpawnedGpuKernel> = inflight job (pack, descriptors, fence).
 *   Caller owns the pointer until join/destroy.
 *
 * #### Throws:
 * - runtime_error / type_error style errors if Context is not ready, meta is
 *   incomplete, arity mismatches, cache/SPIR-V is missing, or Vulkan submit fails.
 *
 * #### Example meta shape (saxpy-style @Gpu):
 * ``py
 * {
 *     "symbol": "saxpy",              # ShaderCache key / kernel name
 *     "binding_count": 3,             # STORAGE_BUFFER bindings (0 scalars + 2 lists)
 *     "scalar_bytes": 8,              # std430 scalar SSBO size (e.g. int n + float a)
 *     "local_size_x": 64,             # compute workgroup size (shader layout)
 *     "group_count_x": None,          # optional override; else ceil(n / local_size_x)
 *     "group_count_y": 1,
 *     "group_count_z": 1,
 *     "params": [
 *         {
 *             "name": "n",
 *             "kind": "int",           # packed into scalar SSBO (binding 0)
 *             "pass_as": "value",
 *         },
 *         {
 *             "name": "a",
 *             "kind": "float",
 *             "pass_as": "value",
 *         },
 *         {
 *             "name": "x",
 *             "kind": "list",          # binding 1
 *             "pass_as": "ref",
 *             "elem_kind": "float",
 *             "elem_bytes": 4,
 *         },
 *         {
 *             "name": "y",
 *             "kind": "list",          # binding 2
 *             "pass_as": "ref",
 *             "elem_kind": "float",
 *             "elem_bytes": 4,
 *         },
 *     ],
 *     "types": {},                    # optional: Python classes for writeback
 *     "schemas": {},                  # optional: field layouts for Threadables
 * }
 * ``
 * ordered_values for that meta would be like ``[n, a, x_list, y_list]``.
 * Scalars are flattened into one SSBO in param order; each list gets its own
 * binding 1..N. Smoke tests may omit types/schemas and pass a smaller dict.
 *
 * #### Technical terms:
 * - spawn_from_meta: CPU launcher that builds SpawnedKernel + CThread/pool task.
 * - Shader cache: process map of symbol -> reusable pipeline and set layout.
 * - Descriptor set: per-launch binding table from GpuPack buffers to the shader.
 */
std::shared_ptr<SpawnedGpuKernel> launch_gpu_kernel(
    py::dict meta,
    py::list ordered_values
);

} // namespace cthreads::gpu
