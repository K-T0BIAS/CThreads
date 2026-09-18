"""
GpuArena: bind Python lists into process GpuState for launch reuse.

Option B launch style: pass the same list objects to `gpu()` after bind.
Launch checks id + length; resident buffers skip alloc/upload. Host edits
between syncs are the caller's responsibility (no proxy yet).
"""

from __future__ import annotations

import struct
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from . import _ext_gpu_api
from .frontend.errors import GPUNotAvailable, GpuInvalidArgument

_ELEM_BYTES: dict[str, int] = {
    "bool": 4,
    "int": 4,
    "float": 4,
    "double": 8,
}

# id(list) -> BoundSlot for every live arena bind (process-wide Option B lookup).
_ID_TO_SLOT: dict[int, "BoundSlot"] = {}
_ID_LOCK = threading.Lock()


@dataclass
class BoundSlot:
    """One arena-bound Python list and its GpuState name."""

    arena_id: str
    name: str
    state_name: str
    host: list[Any]
    numel: int
    elem_kind: str
    elem_bytes: int


def lookup_resident(value: Any) -> BoundSlot | None:
    """
    Return the BoundSlot for a Python list if it is currently arena-bound.

    #### Args:
    - value: Any = launch argument (typically a list)

    #### Returns
    - BoundSlot | None = slot when id(value) is registered
    """
    if not isinstance(value, list):
        return None
    with _ID_LOCK:
        return _ID_TO_SLOT.get(id(value))


def infer_elem_kind(values: list[Any]) -> str:
    """
    Infer GPU list elem_kind from the first element (bool before int).

    #### Args:
    - values: list[Any] = non-empty host list

    #### Returns
    - str = "bool" | "int" | "float" | "double"

    #### Raises
    - GpuInvalidArgument = empty list or unsupported element type
    """
    if not values:
        raise GpuInvalidArgument(
            "GpuArena.bind: cannot infer elem type from an empty list "
            "(pass a non-empty list)"
        )
    sample: Any = values[0]
    if isinstance(sample, bool):
        return "bool"
    if isinstance(sample, int):
        return "int"
    if isinstance(sample, float):
        return "float"
    raise GpuInvalidArgument(
        f"GpuArena.bind: unsupported list element type {type(sample)!r}"
    )


def list_to_bytes(values: list[Any], elem_kind: str) -> bytes:
    """Pack a Python list into std430-friendly host bytes."""
    if elem_kind == "float":
        return struct.pack(f"{len(values)}f", *[float(v) for v in values])
    if elem_kind == "double":
        return struct.pack(f"{len(values)}d", *[float(v) for v in values])
    if elem_kind == "int":
        return struct.pack(f"{len(values)}i", *[int(v) for v in values])
    if elem_kind == "bool":
        return struct.pack(
            f"{len(values)}i", *[1 if bool(v) else 0 for v in values]
        )
    raise GpuInvalidArgument(f"unsupported elem_kind: {elem_kind!r}")


def bytes_into_list(data: bytes, values: list[Any], elem_kind: str) -> None:
    """Write downloaded bytes back into the same Python list object."""
    n: int = len(values)
    if elem_kind == "float":
        unpacked = struct.unpack(f"{n}f", data)
        for i, v in enumerate(unpacked):
            values[i] = float(v)
        return
    if elem_kind == "double":
        unpacked = struct.unpack(f"{n}d", data)
        for i, v in enumerate(unpacked):
            values[i] = float(v)
        return
    if elem_kind == "int":
        unpacked = struct.unpack(f"{n}i", data)
        for i, v in enumerate(unpacked):
            values[i] = int(v)
        return
    if elem_kind == "bool":
        unpacked = struct.unpack(f"{n}i", data)
        for i, v in enumerate(unpacked):
            values[i] = bool(v)
        return
    raise GpuInvalidArgument(f"unsupported elem_kind: {elem_kind!r}")


class GpuArena:
    """
    Session that keeps named list buffers resident in GpuState.

    #### Example:
    ``py
    with GpuArena() as arena:
        arena.bind(x=x, y=y)
        for _ in range(100):
            gpu(saxpy, n, 2.0, x, y).join(download=False)
        arena.sync()
    ``
    """

    def __init__(self) -> None:
        self._id: str = uuid.uuid4().hex
        self._slots: dict[str, BoundSlot] = {}
        self._released: bool = False

    def __enter__(self) -> "GpuArena":
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def bind(self, **named_lists: list[Any]) -> "GpuArena":
        """
        Allocate/upload device buffers for the given lists (by kwarg name).

        Reuses GpuState entries when the same slot name is rebound with the
        same length and elem kind; otherwise removes and recreates.

        #### Args:
        - **named_lists: list[Any] = keyword slot name -> Python list object

        #### Returns
        - GpuArena = this arena (for chaining)

        #### Raises
        - GPUNotAvailable = GPU extension missing
        - GpuInvalidArgument = bad args, empty lists, duplicate list ids
        """
        if self._released:
            raise GpuInvalidArgument("GpuArena.bind: arena already released")
        if not named_lists:
            raise GpuInvalidArgument("GpuArena.bind: expected at least one list")
        if not _ext_gpu_api.available():
            raise GPUNotAvailable(
                "GPU is not available (build with CTHREADS_GPU=ON and a Vulkan device)"
            )

        state = _ext_gpu_api.gpu_state()
        _ext_gpu_api.init()

        for slot_name, host in named_lists.items():
            if not isinstance(host, list):
                raise GpuInvalidArgument(
                    f"GpuArena.bind: {slot_name!r} must be a list, got {type(host)!r}"
                )
            elem_kind: str = infer_elem_kind(host)
            elem_bytes: int = _ELEM_BYTES[elem_kind]
            numel: int = len(host)
            nbytes: int = numel * elem_bytes
            state_name: str = f"{self._id}/{slot_name}"
            host_id: int = id(host)

            with _ID_LOCK:
                existing = _ID_TO_SLOT.get(host_id)
                if existing is not None and (
                    existing.arena_id != self._id or existing.name != slot_name
                ):
                    raise GpuInvalidArgument(
                        f"GpuArena.bind: list for {slot_name!r} is already bound "
                        f"as {existing.arena_id}/{existing.name}"
                    )

            # Replace prior slot with the same name if shape changed.
            prior = self._slots.get(slot_name)
            if prior is not None:
                if (
                    prior.host is host
                    and prior.numel == numel
                    and prior.elem_kind == elem_kind
                ):
                    state.upload(state_name, list_to_bytes(host, elem_kind))
                    continue
                self._unbind_slot(prior, state)

            if state.contains(state_name):
                state.remove(state_name)
            state.add(state_name, nbytes)
            state.upload(state_name, list_to_bytes(host, elem_kind))

            slot = BoundSlot(
                arena_id=self._id,
                name=slot_name,
                state_name=state_name,
                host=host,
                numel=numel,
                elem_kind=elem_kind,
                elem_bytes=elem_bytes,
            )
            self._slots[slot_name] = slot
            with _ID_LOCK:
                _ID_TO_SLOT[host_id] = slot
        return self

    def sync(self, *names: str) -> None:
        """
        Download resident buffers into the bound Python lists.

        #### Args:
        - *names: str = optional slot names; default = all bound slots

        #### Raises
        - GpuInvalidArgument = unknown name or arena released
        """
        if self._released:
            raise GpuInvalidArgument("GpuArena.sync: arena already released")
        state = _ext_gpu_api.gpu_state()
        targets: list[BoundSlot]
        if names:
            targets = []
            for name in names:
                slot = self._slots.get(name)
                if slot is None:
                    raise GpuInvalidArgument(
                        f"GpuArena.sync: unknown slot {name!r}"
                    )
                targets.append(slot)
        else:
            targets = list(self._slots.values())

        for slot in targets:
            if len(slot.host) != slot.numel:
                raise GpuInvalidArgument(
                    f"GpuArena.sync: list for {slot.name!r} changed length "
                    f"(was {slot.numel}, now {len(slot.host)}); rebind"
                )
            data: bytes = bytes(state.download(slot.state_name))
            bytes_into_list(data, slot.host, slot.elem_kind)

    def release(self) -> None:
        """Destroy all GpuState buffers owned by this arena and drop id map entries."""
        if self._released:
            return
        state = None
        try:
            if _ext_gpu_api.available():
                state = _ext_gpu_api.gpu_state()
        except Exception:
            state = None
        for slot in list(self._slots.values()):
            self._unbind_slot(slot, state)
        self._slots.clear()
        self._released = True

    def _unbind_slot(self, slot: BoundSlot, state: Any) -> None:
        with _ID_LOCK:
            cur = _ID_TO_SLOT.get(id(slot.host))
            if cur is slot:
                del _ID_TO_SLOT[id(slot.host)]
        if state is not None and state.contains(slot.state_name):
            try:
                state.remove(slot.state_name)
            except Exception:
                pass
        self._slots.pop(slot.name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def names(self) -> list[str]:
        """Registered slot names in this arena."""
        return list(self._slots.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)
