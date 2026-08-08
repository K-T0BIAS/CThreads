import inspect
from ..CONFIG import VERSION, REGISTRY
from .lib import is_threadable
from typing import Any, get_type_hints


def Threadable(cls):
    """
    Wrapper to mark a class as a @Threadable class

    A class marked with this decorator will be compiled into a c++ struct with the same name.

    #### Rules for @Threadable classes:
    - All methods must be @Threaded
    - All methods must have type hints that are @Threadable
    - All methods must return a @Threadable or basic python type (int, float, str, bool)
    - All methods must accept @Threadable or basic python types as arguments
    - The class must not have an __init__ method

    #### Example:

    >>>> @Threadable
    ... class MyThreadable:
    ...     x: float
    ...     y: float     
    ...     @Thread
    ...     def add(self, x: float, y: float) -> float:
    ...         return self.x + self.y
    """
    cls.__threadable = True
    cls.__threadable_version = VERSION

    anno_dict: dict[str, Any] = get_type_hints(cls)
    for hint in anno_dict.values():
        try:
            is_threadable(hint)
        except TypeError as e:
            raise TypeError(
                f"Threadable class {cls.__name__} has invalid type {hint}"
            ) from e

    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
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

    REGISTRY.register_threadable(cls)
    return cls
