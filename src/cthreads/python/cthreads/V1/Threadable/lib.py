"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, get_args, get_origin

TYPE_WHITELIST = [
    int,
    float,
    str,
    bool,
    list,
    dict,
    set,
]

def is_internal_cthreads_type(item: Any) -> bool:
    """
    Check if the type is an internal CThreads type.
    """
    return isinstance(item, type) and getattr(item, "__cthreads_internal__", False)

def is_threadable(item: Any) -> bool:
    """
    Check if the type hint is allowed in a Threadable class.

    Allowed: whitelist types, generics of those (e.g. list[int]), and @Threadable classes.
    Non-threadable classes and any other type raise TypeError.

    Args:
        item: a type hint

    Returns:
        bool: True if allowed, raises TypeError otherwise
    """
    origin = get_origin(item)
    if origin is not None:
        if origin not in TYPE_WHITELIST:
            raise TypeError(f"Type {item} is not allowed in a Threadable class")
        for arg in get_args(item):
            is_threadable(arg)
        return True

    if item in TYPE_WHITELIST:
        return True
    if getattr(item, "__cthreads_tbuffer__", False):
        inner = getattr(item, "__cthreads_tbuffer_inner__", None)
        if inner is None:
            args = get_args(item)
            if args:
                inner = args[0]
        if inner is not None:
            is_threadable(inner)
        return True
    if is_internal_cthreads_type(item):
        return True
    if isinstance(item, type) and getattr(item, "__threadable", False):
        return True
    raise TypeError(f"Type {item} is not allowed in a Threadable class")
