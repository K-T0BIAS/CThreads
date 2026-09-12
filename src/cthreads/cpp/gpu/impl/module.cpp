#include "../headers/module.hpp"
#include "../headers/context.hpp"
#include "../headers/shader.hpp"
#include "../headers/shader_cache.hpp"
#include "../headers/pack.hpp"
#include "../headers/descriptors.hpp"

#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/gil.h>

namespace cthreads::gpu {
namespace {

// Host byte sizes for the scalar SSBO (std430 / SPIR-V). Not C++ sizeof for
// bool/int: GLSL bool is stored as 32-bit; use int32_t so Win/Linux match.
// string is not a scalar-SSBO type on the GPU path.
static const std::unordered_map<std::string, size_t> py_size_of = {
    {"bool", 4},    // int32 0/1
    {"int", 4},     // int32_t
    {"float", 4},   // float
    {"double", 8},  // float64; align 8 when laying out
};

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t std430_align_of(const std::string& kind) {
    // std430: scalar alignment equals its size for these types.
    return py_size_of.at(kind);
}

void release_inflight(Context& context, SpawnedGpuKernel& job) {
    // Destroying the pool frees any CBs allocated from it; free first when we can.
    if (job.command_buffer != VK_NULL_HANDLE &&
        job.command_pool != VK_NULL_HANDLE &&
        context.device != VK_NULL_HANDLE &&
        context.vkFreeCommandBuffers) { // free the cmd buffer when all relevant ressources are valid
        context.vkFreeCommandBuffers(
            context.device, job.command_pool, 1, &job.command_buffer);
    }
    job.command_buffer = VK_NULL_HANDLE;

    // Per-launch command pool (not the TransferEngine pool).
    if (job.command_pool != VK_NULL_HANDLE &&
        context.device != VK_NULL_HANDLE &&
        context.vkDestroyCommandPool) {
        context.vkDestroyCommandPool(context.device, job.command_pool, nullptr);
    }
    job.command_pool = VK_NULL_HANDLE;

    if (job.descriptor_set != VK_NULL_HANDLE) { // free the descriptors (if not freed yet)
        pack::free_set(context, job.descriptor_pool, job.descriptor_set);
    }
    pack::destroy_pool(context, job.descriptor_pool);
    // destroy the fence if not already done
    if (job.fence != VK_NULL_HANDLE && context.device != VK_NULL_HANDLE &&
        context.vkDestroyFence) {
        context.vkDestroyFence(context.device, job.fence, nullptr);
    }
    job.fence = VK_NULL_HANDLE;
    // destroy the pack
    pack::destroy_gpu_pack(context, job.pack);
    job.symbol.clear(); // clear the symbol
    job.writeback_lists.clear();
    job.values_keep.reset();
}

void mark_done(SpawnedGpuKernel& job) {
    { // lock the done_mu mutex and set the done flag
        std::lock_guard<std::mutex> lock(job.done_mu);
        job.done_flag = true;
    }
    job.done_cv.notify_all(); // notify all waiting threads (main thread currently, in later version also cthreads)
}

// After dispatch fence: make SHADER_WRITE visible to TRANSFER_READ for downloads.
// Uses TransferEngine pool/fence under transfer_engine_mutex (same as uploads).
void compute_to_transfer_barrier(Context& context) {
    std::lock_guard<std::mutex> lock(context.transfer_engine_mutex);
    if (context.transfer_engine.command_pool == VK_NULL_HANDLE ||
        context.transfer_engine.fence == VK_NULL_HANDLE ||
        !context.vkCmdPipelineBarrier || !context.vkAllocateCommandBuffers ||
        !context.vkBeginCommandBuffer || !context.vkEndCommandBuffer ||
        !context.vkQueueSubmit || !context.vkResetFences ||
        !context.vkWaitForFences || !context.vkFreeCommandBuffers ||
        !context.queue) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier missing "
            "TransferEngine or entry points");
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = context.transfer_engine.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (context.vkAllocateCommandBuffers(context.device, &alloc_info, &cmd) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkAllocateCommandBuffers failed");
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (context.vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(
            context.device, context.transfer_engine.command_pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkBeginCommandBuffer failed");
    }

    VkMemoryBarrier mem_barrier{};
    mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    context.vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &mem_barrier,
        0,
        nullptr,
        0,
        nullptr);

    if (context.vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(
            context.device, context.transfer_engine.command_pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkEndCommandBuffer failed");
    }

    VkFence te_fence = context.transfer_engine.fence;
    if (context.vkResetFences(context.device, 1, &te_fence) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(
            context.device, context.transfer_engine.command_pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkResetFences failed");
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (context.vkQueueSubmit(context.queue, 1, &submit, te_fence) !=
        VK_SUCCESS) {
        context.vkFreeCommandBuffers(
            context.device, context.transfer_engine.command_pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkQueueSubmit failed");
    }
    if (context.vkWaitForFences(
            context.device, 1, &te_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        context.vkFreeCommandBuffers(
            context.device, context.transfer_engine.command_pool, 1, &cmd);
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: compute_to_transfer_barrier "
            "vkWaitForFences failed");
    }
    context.vkFreeCommandBuffers(
        context.device, context.transfer_engine.command_pool, 1, &cmd);
}

// Download each ref list SSBO into the kept Python list (in place).
void writeback_ref_lists(Context& context, SpawnedGpuKernel& job) {
    if (job.writeback_lists.empty()) {
        return;
    }
    if (!job.values_keep) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: join writeback missing values_keep");
    }

    py::gil_scoped_acquire gil;
    py::list& values = *job.values_keep;

    for (const SpawnedGpuKernel::WritebackListSlot& slot : job.writeback_lists) {
        if (slot.numel == 0) {
            continue;
        }
        if (slot.value_index >= static_cast<size_t>(values.size())) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: writeback value_index out of "
                "range");
        }
        py::list list_val = values[slot.value_index].cast<py::list>();
        if (static_cast<size_t>(list_val.size()) != slot.numel) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: writeback list length changed "
                "during job");
        }

        if (slot.elem_kind == "float") {
            std::vector<float> host(slot.numel);
            pack::download_container(
                context,
                job.pack,
                slot.container_index,
                host.data(),
                host.size() * sizeof(float));
            for (size_t j = 0; j < slot.numel; ++j) {
                list_val[j] = host[j];
            }
        } else if (slot.elem_kind == "int") {
            std::vector<std::int32_t> host(slot.numel);
            pack::download_container(
                context,
                job.pack,
                slot.container_index,
                host.data(),
                host.size() * sizeof(std::int32_t));
            for (size_t j = 0; j < slot.numel; ++j) {
                list_val[j] = host[j];
            }
        } else if (slot.elem_kind == "double") {
            std::vector<double> host(slot.numel);
            pack::download_container(
                context,
                job.pack,
                slot.container_index,
                host.data(),
                host.size() * sizeof(double));
            for (size_t j = 0; j < slot.numel; ++j) {
                list_val[j] = host[j];
            }
        } else {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: unsupported writeback "
                "elem_kind: " +
                slot.elem_kind);
        }
    }
}

// Write one Python scalar into the host scalar blob at offset (std430 layout).
void write_scalar_bytes(
    std::vector<std::uint8_t>& blob,
    size_t offset,
    const std::string& kind,
    const py::object& value
) {
    if (kind == "int") {
        const std::int32_t v = value.cast<std::int32_t>();
        std::memcpy(blob.data() + offset, &v, sizeof(v));
    } else if (kind == "float") {
        const float v = value.cast<float>();
        std::memcpy(blob.data() + offset, &v, sizeof(v));
    } else if (kind == "double") {
        const double v = value.cast<double>();
        std::memcpy(blob.data() + offset, &v, sizeof(v));
    } else if (kind == "bool") {
        // GLSL bool in std430 is 32-bit; store 0/1 as int32.
        const std::int32_t v = value.cast<bool>() ? 1 : 0;
        std::memcpy(blob.data() + offset, &v, sizeof(v));
    } else {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: cannot pack scalar kind: " + kind);
    }
}

} // namespace

void SpawnedGpuKernel::start() {
    // Default path submits at launch time; start is a shared API no-op.
}

void SpawnedGpuKernel::wait() {
    std::unique_lock<std::mutex> lock(done_mu);
    done_cv.wait(lock, [this] { return done_flag; });
}

bool SpawnedGpuKernel::done() {
    std::lock_guard<std::mutex> lock(done_mu);
    return done_flag;
}

void SpawnedGpuKernel::join(Context& context) {
    if (finished) {
        if (eptr) {
            std::rethrow_exception(eptr);
        }
        return;
    }

    try {
        if (fence != VK_NULL_HANDLE) {
            if (!context.ready || context.device == VK_NULL_HANDLE ||
                !context.vkWaitForFences) {
                throw std::runtime_error(
                    "cthreads.gpu.VulkanInitFailed: SpawnedGpuKernel::join "
                    "needs a ready device and vkWaitForFences");
            }
            // wait for the fence to be signaled
            if (context.vkWaitForFences(
                    context.device, 1, &fence, VK_TRUE, UINT64_MAX) !=
                VK_SUCCESS) {
                throw std::runtime_error(
                    "cthreads.gpu.VulkanInitFailed: vkWaitForFences failed in "
                    "SpawnedGpuKernel::join");
            }
        }

        // Permanent list writeback path (Threadable/schema marshal is later).
        if (!writeback_lists.empty()) {
            compute_to_transfer_barrier(context);
            writeback_ref_lists(context, *this);
        }

        mark_done(*this); // mark the job as done
        release_inflight(context, *this); // release the inflight GPU state (clears all buffers, fences and cmd structures)
        finished = true;
    } catch (...) {
        eptr = std::current_exception();
        mark_done(*this);
        try {
            release_inflight(context, *this);
        } catch (...) {
            // Prefer the original join error.
        }
        finished = true;
        std::rethrow_exception(eptr);
    }

    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

SpawnedGpuKernel::~SpawnedGpuKernel() {
    if (finished) {
        return;
    }
    try {
        // try to await the fence and clear the inflight GPU state before destroying the object
        Context& ctx = context();
        if (ctx.ready && ctx.device != VK_NULL_HANDLE) {
            if (fence != VK_NULL_HANDLE && ctx.vkWaitForFences) {
                ctx.vkWaitForFences(
                    ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
            }
            release_inflight(ctx, *this);
        }
    } catch (...) {
        // Destructor must not throw.
    }
    mark_done(*this); // mark the job as done (doesnt mean the job finished successfully, just means it was terminated)
    finished = true;
}

std::shared_ptr<SpawnedGpuKernel> launch_gpu_kernel(
    py::dict meta,
    py::list ordered_values
) {
    // Launch path:
    //   Python args + gpu meta 
    //   -> create/fill GpuPack (upload via staging)
    //   -> ShaderCache.get (create_entry/add is registry-only; missing => throw)
    //   -> allocate descriptor set, update_descriptors(pack)
    //   -> record CB: barrier, bind pipeline, bind set, dispatch
    //   -> create fence, vkQueueSubmit(..., fence)  [no wait here]
    //   -> stash handles on SpawnedGpuKernel (pack, set, pool, CB, fence, symbol, groups)
    //   -> return SpawnedGpuKernel  (no CThread)
    // join(context): wait fence -> download/writeback -> release_inflight

    // Ensure Vulkan Context is ready (loader + device + TransferEngine).
    init(); // init the context (noop if already initialized)
    Context& context = cthreads::gpu::context();
    if (!context.ready) {
        throw std::runtime_error(
            "cthreads.gpu.VulkanInitFailed: launch_gpu_kernel needs a ready "
            "Context");
    }

    if (!meta.contains("symbol") || !meta.contains("params")) {
        throw std::runtime_error(
            "cthreads.gpu.GpuInvalidArgument: launch_gpu_kernel meta needs "
            "'symbol' and 'params'");
    }

    const std::string symbol = meta["symbol"].cast<std::string>(); // kernel / fn name
    py::list params = meta["params"]; // list of input params (see docs string for example)
    // ensure this fn call mathces the number of params in the kernels meta
    if (static_cast<size_t>(ordered_values.size()) != static_cast<size_t>(params.size())) {
        throw py::type_error(
            "cthreads.gpu: expected " + std::to_string(params.size()) +
            " args for '" + symbol + "', got " +
            std::to_string(ordered_values.size()));
    }

    // Walk params: scalars -> binding 0 blob; lists -> ContainerSpec (bindings 1..N).
    // build the container specs + scalar layout (std430 align while summing)
    std::vector<pack::ContainerSpec> container_specs;
    std::vector<std::uint8_t> scalar_host; // filled after we know scalar_bytes
    struct ScalarSlot { // private helper struct to record metadata of scalar params in the kernel call
        size_t value_index = 0;
        size_t offset = 0;
        std::string kind;
    };
    std::vector<ScalarSlot> scalar_slots;
    struct ContainerSlotPlan { // private helper struct to record metadata of list params in the kernel call
        size_t value_index = 0;
        size_t elem_bytes = 0;
        size_t numel = 0;
        std::string elem_kind;
        bool writeback = true; // pass_as ref (default for lists)
    };
    std::vector<ContainerSlotPlan> container_plans;

    size_t scalar_bytes = 0;
    // iter all the params and build the container specs + scalar layout (std430 align while summing)
    for (size_t i = 0; i < static_cast<size_t>(params.size()); ++i) {
        // each param is a dict from meta (see launch_gpu_kernel docstring example)
        py::dict param = params[i].cast<py::dict>(); // this param is a dict with meta like name, kind, pass_as, numel, elem_kind, elem_bytes, ...
        std::string name = param["name"].cast<std::string>(); // the var name / identifier (set by usr code)
        std::string kind = param["kind"].cast<std::string>(); // the type
        const bool is_list = (kind == "list");
        // Scalars default value; lists default ref (join writeback).
        std::string pass_as = param.contains("pass_as")
            ? param["pass_as"].cast<std::string>()
            : (is_list ? std::string("ref") : std::string("value"));
        (void)name;

        if (is_list) { // lists get theri own ssbo (single buffer)
            // List SSBO: elem size from meta; numel from the Python list length.
            std::string elem_kind = param.contains("elem_kind") //type (default float)
                ? param["elem_kind"].cast<std::string>()
                : std::string("float");
            size_t elem_bytes = 0;
            // check if the bytes are set in the meta, otherwise try to infer from the type or throw an err
            if (param.contains("elem_bytes")) {
                elem_bytes = param["elem_bytes"].cast<size_t>();
            } else if (py_size_of.find(elem_kind) != py_size_of.end()) {
                elem_bytes = py_size_of.at(elem_kind);
            } else {
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: unknown list elem type: " +
                    elem_kind);
            }
            // Lists default to pass_as ref (in-place writeback on join).
            if (pass_as != "ref" && pass_as != "value") {
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: unsupported pass_as for "
                    "list '" +
                    name + "': " + pass_as);
            }
            const bool do_writeback = (pass_as != "value");
            // gte the actuall value from the kernel call (py side)
            py::list list_val = ordered_values[i].cast<py::list>();
            const size_t numel = static_cast<size_t>(list_val.size());
            // optional meta numel must match the live list if both present
            if (param.contains("numel") && param["numel"].cast<size_t>() != numel) { // check that sizes match the expectations form the meta
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: meta numel does not match "
                    "list length for '" +
                    name + "'");
            }
            container_specs.push_back(pack::ContainerSpec{elem_bytes, numel});
            container_plans.push_back(ContainerSlotPlan{
                i, elem_bytes, numel, std::move(elem_kind), do_writeback});
            continue;
        }
        // --- handle scalar args ---

        // Scalar field into the binding-0 SSBO (not a separate descriptor).
        if (py_size_of.find(kind) == py_size_of.end()) { // check if we can interpret this type
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: unknown variable type: " +
                kind);
        }
        if (pass_as != "value" && pass_as != "ref") {
            // Scalars are always packed by value into the SSBO; pass_as is
            // reserved for future semantics. Reject unknown tags early.
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: unsupported pass_as for "
                "scalar '" +
                name + "': " + pass_as);
        }
        const size_t size = py_size_of.at(kind);
        const size_t alignment = std430_align_of(kind);
        scalar_bytes = align_up(scalar_bytes, alignment);
        scalar_slots.push_back(ScalarSlot{i, scalar_bytes, kind});
        scalar_bytes += size;
    }

    // Prefer meta scalar_bytes when present (codegen truth); must match walk.
    if (meta.contains("scalar_bytes")) {
        const size_t meta_bytes = meta["scalar_bytes"].cast<size_t>();
        if (meta_bytes != scalar_bytes) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: meta scalar_bytes (" +
                std::to_string(meta_bytes) + ") != layout sum (" +
                std::to_string(scalar_bytes) + ")");
        }
    }

    // Job owns GPU objects from here on so failures can release_inflight.
    auto job = std::make_shared<SpawnedGpuKernel>();
    job->symbol = symbol;
    // Keep the same Python arg objects for join writeback (list identity).
    job->values_keep = std::make_shared<py::list>(ordered_values);
    for (size_t c = 0; c < container_plans.size(); ++c) {
        const ContainerSlotPlan& plan = container_plans[c];
        if (!plan.writeback || plan.numel == 0) {
            continue;
        }
        job->writeback_lists.push_back(SpawnedGpuKernel::WritebackListSlot{
            plan.value_index,
            c,
            plan.numel,
            plan.elem_kind,
        });
    }
    // collect launch group data (required to ensure the correct num threads a re launched and the correct thread block shape is used)
    if (meta.contains("group_count_x") && !meta["group_count_x"].is_none()) {
        job->group_count_x = meta["group_count_x"].cast<uint32_t>();
    }
    if (meta.contains("group_count_y") && !meta["group_count_y"].is_none()) {
        job->group_count_y = meta["group_count_y"].cast<uint32_t>();
    }
    if (meta.contains("group_count_z") && !meta["group_count_z"].is_none()) {
        job->group_count_z = meta["group_count_z"].cast<uint32_t>();
    }

    try {
        // init the gpu pack (device-local scalar blob + one buffer per list)
        job->pack = pack::create_gpu_pack(
            context,
            scalar_bytes,
            container_specs
        );

        // Pack Python scalars into a host byte blob, then upload through staging.
        if (scalar_bytes > 0) {
            scalar_host.assign(scalar_bytes, 0);
            for (const ScalarSlot& slot : scalar_slots) {
                write_scalar_bytes(
                    scalar_host,
                    slot.offset,
                    slot.kind,
                    ordered_values[slot.value_index].cast<py::object>());
            }
            // upload the scalars
            pack::upload_scalars(
                context, job->pack, scalar_host.data(), scalar_bytes);
        }

        // Upload each list container (binding 1..N) from ordered_values.
        for (size_t c = 0; c < container_plans.size(); ++c) {
            const ContainerSlotPlan& plan = container_plans[c];
            if (plan.numel == 0) {
                continue; // empty slot: no VkBuffer; update_descriptors still rejects empty for now
            }
            py::list list_val = ordered_values[plan.value_index].cast<py::list>(); // get the py side list that was passed in the kernel call
            if (plan.elem_kind == "float") {
                std::vector<float> host(plan.numel);
                for (size_t j = 0; j < plan.numel; ++j) {
                    host[j] = list_val[j].cast<float>();
                }
                pack::upload_container(
                    context,
                    job->pack,
                    c,
                    host.data(),
                    host.size() * sizeof(float));
            } else if (plan.elem_kind == "int") {
                std::vector<std::int32_t> host(plan.numel);
                for (size_t j = 0; j < plan.numel; ++j) {
                    host[j] = list_val[j].cast<std::int32_t>();
                }
                pack::upload_container(
                    context,
                    job->pack,
                    c,
                    host.data(),
                    host.size() * sizeof(std::int32_t));
            } else if (plan.elem_kind == "double") {
                std::vector<double> host(plan.numel);
                for (size_t j = 0; j < plan.numel; ++j) {
                    host[j] = list_val[j].cast<double>();
                }
                pack::upload_container(
                    context,
                    job->pack,
                    c,
                    host.data(),
                    host.size() * sizeof(double));
            } else {
                throw std::runtime_error(
                    "cthreads.gpu.GpuInvalidArgument: unsupported list elem_kind: " +
                    plan.elem_kind);
            }
        }

        // get the shader cache entry (must already be registered)
        const shader::ShaderCacheEntry& entry =
            shader::ShaderCache::getInstance().get(symbol);

        // binding_count on the entry must match 1 + number of list slots
        const uint32_t expected_bindings =
            1u + static_cast<uint32_t>(container_specs.size());
        if (entry.binding_count != expected_bindings) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: ShaderCacheEntry binding_count (" +
                std::to_string(entry.binding_count) + ") != 1 + list count (" +
                std::to_string(expected_bindings) + ")");
        }
        if (entry.pipeline == VK_NULL_HANDLE ||
            entry.pipeline_layout == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "cthreads.gpu.GpuInvalidArgument: ShaderCacheEntry missing "
                "pipeline for '" +
                symbol + "'");
        }

        // get the descriptor pool to allocate the descriptor set next
        job->descriptor_pool =
            pack::create_pool(context, entry.binding_count, 1);
        // allocate the descriptor set
        job->descriptor_set =
            pack::allocate_set(context, job->descriptor_pool, entry.set_layout);
        // wire binding i -> pack buffer i (schema from entry, buffers from this pack)
        pack::update_descriptors(
            context, job->descriptor_set, entry, job->pack);

        // Need bind/dispatch/barrier + the usual CB/submit PFNs.
        if (!context.vkCreateCommandPool || !context.vkDestroyCommandPool ||
            !context.vkAllocateCommandBuffers || !context.vkFreeCommandBuffers ||
            !context.vkBeginCommandBuffer || !context.vkEndCommandBuffer ||
            !context.vkCmdPipelineBarrier || !context.vkCmdBindPipeline ||
            !context.vkCmdBindDescriptorSets || !context.vkCmdDispatch ||
            !context.vkCreateFence || !context.vkDestroyFence ||
            !context.vkQueueSubmit || !context.queue) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: launch_gpu_kernel missing "
                "dispatch/command/fence entry points or queue");
        }

        // Per-job command pool: own lifetime, no TransferEngine mutex needed.
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = context.queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (context.vkCreateCommandPool(
                context.device, &pool_info, nullptr, &job->command_pool) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateCommandPool failed in "
                "launch_gpu_kernel");
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = job->command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        if (context.vkAllocateCommandBuffers(
                context.device, &alloc_info, &job->command_buffer) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkAllocateCommandBuffers failed "
                "in launch_gpu_kernel");
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (context.vkBeginCommandBuffer(job->command_buffer, &begin_info) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkBeginCommandBuffer failed in "
                "launch_gpu_kernel");
        }

        // Uploads already waited on the TransferEngine fence, but Vulkan still
        // needs a barrier so compute sees TRANSFER_WRITE results.
        VkMemoryBarrier mem_barrier{};
        mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mem_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mem_barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        context.vkCmdPipelineBarrier(
            job->command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &mem_barrier,
            0,
            nullptr,
            0,
            nullptr);

        // Bind compute pipeline + this launch's descriptor set, then dispatch.
        context.vkCmdBindPipeline(
            job->command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            entry.pipeline);
        context.vkCmdBindDescriptorSets(
            job->command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            entry.pipeline_layout,
            0,
            1,
            &job->descriptor_set,
            0,
            nullptr);
        context.vkCmdDispatch(
            job->command_buffer,
            job->group_count_x,
            job->group_count_y,
            job->group_count_z);

        if (context.vkEndCommandBuffer(job->command_buffer) != VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkEndCommandBuffer failed in "
                "launch_gpu_kernel");
        }

        // Per-job fence (not TransferEngine.fence). Unsignaled until submit done.
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (context.vkCreateFence(
                context.device, &fence_info, nullptr, &job->fence) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkCreateFence failed in "
                "launch_gpu_kernel");
        }

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &job->command_buffer;
        if (context.vkQueueSubmit(
                context.queue, 1, &submit, job->fence) != VK_SUCCESS) {
            throw std::runtime_error(
                "cthreads.gpu.VulkanInitFailed: vkQueueSubmit failed in "
                "launch_gpu_kernel");
        }
        // Do not wait here — join() waits on job->fence.
    } catch (...) {
        // Tear down any handles already stashed; then rethrow to Python.
        try {
            release_inflight(context, *job);
        } catch (...) {
            // Prefer the original launch error.
        }
        throw;
    }

    return job;
}

} // namespace cthreads::gpu
