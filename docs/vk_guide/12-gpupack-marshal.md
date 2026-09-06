# 12 — GpuPack and marshal (option 5 end-to-end)

## Definition

**GpuPack** is the GPU analogue of the CPU args pack for one launch:

```text
GpuPack {
  GpuBuffer scalars;          // DeviceLocal, std430 blob
  vector<GpuBuffer> lists;    // DeviceLocal, one per list arg
}
```

Optional host-side metadata:

- dtype per list (float vs int)
- element counts
- pointer/identity of Python objects for writeback

## Build from Python args (marshal / smoke)

Example call:

```text
n: int = 4
a: float = 2.0
x: list[float] len 4
y: list[float] len 4
```

Steps:

1. Build host `ScalarPack { n, a }` bytes.
2. `scalars = create_buffer(sizeof(ScalarPack), DeviceLocal)`.
3. `upload_buffer(scalars, &host_scalars, sizeof)`.
4. For `x`: create device-local buffer `4*sizeof(float)`; upload from a temporary C array filled from the Python list (or from a contiguous buffer you built while iterating).
5. Same for `y`.
6. Store Python object references for writeback.

Empty list rule: **do not** create a 0-size VkBuffer; store "no buffer / count 0".

## After compute (join)

1. Wait fence.
2. `download_buffer` scalars -> host struct -> write Python ints/floats if they were mutable outputs (usually scalars are inputs; returns go in scalar blob too).
3. `download_buffer` each list -> update the **same** Python list objects in place (like CPU writeback).
4. Destroy pack buffers when the job is done (or retain if you design persistent buffers later — v1 can destroy per job).

## Comparison table

| Concern | CPU | GPU |
|---------|-----|-----|
| Where scalars live | Fields in C++ pack | Scalar SSBO bytes |
| Where lists live | `vector` / container in pack | Per-list SSBO |
| Fill | `pack_params` trampolines | `upload_buffer` |
| Run | call C++ function | dispatch SPIR-V |
| Sync mid-run | optional `__sync_state` | **not in v1** |
| Finish | writeback | download + writeback |

## What marshal must guarantee

1. Host scalar blob matches shader `std430` block.
2. List buffers contain tightly packed elements of the declared dtype.
3. Descriptor bindings match the emitter's binding assignment.
4. Download size matches what was uploaded (or the mutated full buffer).

## Internal test API vs public API

Tests may use an internal helper such as `_ext.gpu.roundtrip_lists(...)`.
That is **not** a user-facing `DeviceBuffer`.

Public surface stays:

```python
from cthreads import gpu
gpu.available()
# later: gpu(fn, *args).join()
```

## Lifetime vs Context

```text
Context          process lifetime (init/shutdown)
TransferEngine   process lifetime (optional reuse)
GpuPack          per job (or per batch)
GpuBuffer        owned by pack / staging temps
```

Destroy packs before shutting down Context.
Destroy transfer engine resources before destroying the device.
