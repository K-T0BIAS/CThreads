"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Host-side opaque handles for ``TBuffer[Threadable]``.

Allocation lives in ``kernels.dll`` (``cthreads_create_tbuffer``). ``_ext`` /
marshal only pass the ``void*`` through to typed kernel params.
"""

from __future__ import annotations

import ctypes
from typing import Any


class TBufferHandle:
    """Opaque host handle to ``tripple_buffer<Threadable>`` in ``kernels.dll``."""

    __slots__ = ("type_name", "ptr", "_destroyed")

    def __init__(self, type_name: str, ptr: int) -> None:
        if not ptr:
            raise ValueError("TBufferHandle: null pointer")
        self.type_name = type_name
        self.ptr = int(ptr)
        self._destroyed = False

    def destroy(self) -> None:
        if self._destroyed or not self.ptr:
            return
        destroy_tbuffer(self)
        self.ptr = 0
        self._destroyed = True

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


def create_tbuffer(threadable_cls: type, capacity: int) -> TBufferHandle:
    """
    Allocate ``tripple_buffer<Threadable>`` in the loaded kernel library.

    Requires ``compile()`` + ``build()`` + ``load_kernels()`` so
    ``cthreads_create_tbuffer`` exists for that Threadable type.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not getattr(threadable_cls, "__threadable", False):
        raise TypeError(f"{threadable_cls!r} is not a @Threadable class")

    from . import marshal

    name = threadable_cls.__name__
    lib = marshal._lib()
    fn = marshal._fn(lib, "cthreads_create_tbuffer")
    ptr = marshal._call(
        fn,
        ctypes.c_void_p,
        [ctypes.c_char_p, ctypes.c_int],
        name.encode("utf-8"),
        int(capacity),
    )
    if not ptr:
        raise RuntimeError(
            f"cthreads.create_tbuffer: failed for {name!r} "
            f"(rebuild kernels after adding TBuffer[{name}])"
        )
    return TBufferHandle(name, int(ptr))


def destroy_tbuffer(handle: TBufferHandle) -> None:
    """Free a handle allocated by ``create_tbuffer``."""
    if handle._destroyed or not handle.ptr:
        return

    from . import marshal

    lib = marshal._lib()
    fn = marshal._fn(lib, "cthreads_destroy_tbuffer")
    marshal._call(
        fn,
        None,
        [ctypes.c_char_p, ctypes.c_void_p],
        handle.type_name.encode("utf-8"),
        ctypes.c_void_p(handle.ptr),
    )
    handle.ptr = 0
    handle._destroyed = True


def tbuffer_ptr(value: Any, inner_type_name: str | None = None) -> int:
    """Resolve a host buffer value to a native pointer for marshal."""
    if isinstance(value, TBufferHandle):
        if inner_type_name is not None and value.type_name != inner_type_name:
            raise TypeError(
                f"TBufferHandle type {value.type_name!r} != "
                f"expected {inner_type_name!r}"
            )
        if value._destroyed or not value.ptr:
            raise RuntimeError("TBufferHandle already destroyed")
        return value.ptr

    import importlib

    _ext = importlib.import_module("cthreads._ext")
    fn = _ext.tbuffer_native_ptr
    if inner_type_name is not None:
        return int(fn(value, inner_type_name))
    return int(fn(value))


def tbuffer_generation(handle: TBufferHandle) -> int:
    from . import marshal

    lib = marshal._lib()
    fn = marshal._fn(lib, "cthreads_tbuffer_generation")
    return int(
        marshal._call(
            fn,
            ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_void_p],
            handle.type_name.encode("utf-8"),
            ctypes.c_void_p(handle.ptr),
        )
    )


def tbuffer_read_copy_ptr(handle: TBufferHandle) -> int:
    """Return native pointer to a heap snapshot of the published slot."""
    from . import marshal

    lib = marshal._lib()
    fn = marshal._fn(lib, "cthreads_tbuffer_read_copy")
    ptr = marshal._call(
        fn,
        ctypes.c_void_p,
        [ctypes.c_char_p, ctypes.c_void_p],
        handle.type_name.encode("utf-8"),
        ctypes.c_void_p(handle.ptr),
    )
    if not ptr:
        raise RuntimeError("cthreads.tbuffer_read_copy: null snapshot")
    return int(ptr)


def tbuffer_free_read_copy(handle: TBufferHandle, copy_ptr: int) -> None:
    if not copy_ptr:
        return
    from . import marshal

    lib = marshal._lib()
    fn = marshal._fn(lib, "cthreads_tbuffer_free_read_copy")
    marshal._call(
        fn,
        None,
        [ctypes.c_char_p, ctypes.c_void_p],
        handle.type_name.encode("utf-8"),
        ctypes.c_void_p(copy_ptr),
    )
