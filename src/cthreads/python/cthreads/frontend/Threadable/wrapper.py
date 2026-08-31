"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect
from ..Registry import REGISTRY
from .lib import is_threadable
from typing import Any, get_origin, get_type_hints

# Signature placeholder for list/dict/set/nested fields (new container each call).
_FACTORY = type("_FACTORY", (), {"__repr__": lambda self: "<factory>"})()


def Threadable(cls):
    """
    Wrapper to mark a class as a @Threadable class

    A class marked with this decorator will be compiled into a c++ struct with the same name.

    #### Rules for @Threadable classes:
    - All methods must be @Threaded
    - All methods must have type hints that are @Threadable
    - All methods must return a @Threadable or basic python type (int, float, str, bool)
    - All methods must accept @Threadable or basic python types as arguments
    - Do not define `__init__`; the decorator supplies a dataclass-style
      constructor (positional/keyword fields; omitted fields match C++ `T{}`)

    #### Example:

    >>>> @Threadable
    ... class MyThreadable:
    ...     x: float
    ...     y: float     
    ...     @Thread
    ...     def add(self, x: float, y: float) -> float:
    ...         return self.x + self.y
    """
    if "__init__" in cls.__dict__:
        raise TypeError(
            f"Threadable class {cls.__name__} must not define __init__; "
            "the decorator supplies a dataclass-style constructor"
        )

    cls.__threadable = True
    cls.__threadable_version = REGISTRY.VERSION

    # localns includes the class so self-refs (list[Boid] / list["Boid"]) resolve
    # while the decorator is still running.
    field_hints: dict[str, Any] = get_type_hints(cls, localns={cls.__name__: cls})
    for hint in field_hints.values():
        try:
            is_threadable(hint)
        except TypeError as e:
            raise TypeError(
                f"Threadable class {cls.__name__} has invalid type {hint}"
            ) from e

    # ensure all methods are @Thread wrapped
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name == "__init__":
            continue
        if not getattr(func, "__threaded", False):
            # plain methods: still require threadable-safe annotations if present
            anno_dict = get_type_hints(func)
            for hint in anno_dict.values():
                try:
                    is_threadable(hint)
                except TypeError as e:
                    raise TypeError(
                        f"Threadable function {name} has invalid type {hint}"
                    ) from e
            continue

        anno_dict = get_type_hints(func)
        for hint_name, hint in anno_dict.items():
            if hint_name == "return" and hint in (None, type(None)):
                continue
            try:
                is_threadable(hint)
            except TypeError as e:
                raise TypeError(
                    f"Threadable method {name} has invalid type {hint}"
                ) from e

    # Host dataclass-style ctor. Omitted fields match C++ `T{}` / `Name() = default`.
    hints = tuple(field_hints.items())
    field_names = tuple(name for name, _ in hints)
    field_set = frozenset(field_names)
    cls_name = cls.__name__

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > len(field_names):
            n = len(field_names)
            raise TypeError(
                f"{cls_name}.__init__() takes from 1 to {n + 1} positional "
                f"arguments but {len(args) + 1} were given"
            )
        assigned = {}
        for i, val in enumerate(args):
            assigned[field_names[i]] = val
        for key, val in kwargs.items():
            if key not in field_set:
                raise TypeError(
                    f"{cls_name}.__init__() got an unexpected keyword argument {key!r}"
                )
            if key in assigned:
                raise TypeError(
                    f"{cls_name}.__init__() got multiple values for argument {key!r}"
                )
            assigned[key] = val
        for field_name, hint in hints:
            if field_name in assigned:
                setattr(self, field_name, assigned[field_name])
                continue
            origin = get_origin(hint)
            if origin is list:
                setattr(self, field_name, [])
            elif origin is dict:
                setattr(self, field_name, {})
            elif origin is set:
                setattr(self, field_name, set())
            elif hint is int:
                setattr(self, field_name, 0)
            elif hint is float:
                setattr(self, field_name, 0.0)
            elif hint is bool:
                setattr(self, field_name, False)
            elif hint is str:
                setattr(self, field_name, "")
            elif isinstance(hint, type):
                try:
                    setattr(self, field_name, hint())
                except TypeError as e:
                    raise TypeError(
                        f"Threadable class {type(self).__name__}: field {field_name!r} "
                        f"type {hint!r} has no zero-arg constructor"
                    ) from e
            else:
                raise TypeError(
                    f"Threadable class {type(self).__name__}: "
                    f"cannot default-construct field {field_name!r} ({hint!r})"
                )

    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for field_name, hint in hints:
        if hint is int:
            default = 0
        elif hint is float:
            default = 0.0
        elif hint is bool:
            default = False
        elif hint is str:
            default = ""
        else:
            default = _FACTORY
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=hint,
            )
        )
    __init__.__signature__ = inspect.Signature(params)
    __init__.__qualname__ = f"{cls_name}.__init__"
    cls.__init__ = __init__
    REGISTRY.register_threadable(cls)
    return cls
