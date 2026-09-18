from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ....types import PyType

# (binding index, param name, kind) — kind is "scalar" or "list"
BindingName = tuple[int, str, str]


@dataclass
class GpuTranslationContext:
    """Mutable codegen state for one @Gpu function."""

    fn: Callable
    local_size_x: int = 64
    symbols: dict[str, PyType] = field(default_factory=dict)
    binding_names: list[BindingName] = field(default_factory=list)
    # Param names that live in the binding-0 Scalars SSBO (not bare GLSL ids).
    scalar_params: set[str] = field(default_factory=set)
    # Param names that are list SSBO instances (x -> x.data[i] via Index).
    list_params: set[str] = field(default_factory=set)

    @property
    def func_name(self) -> str:
        return self.fn.__name__
