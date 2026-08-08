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
    if isinstance(item, type) and getattr(item, "__threadable", False):
        return True
    raise TypeError(f"Type {item} is not allowed in a Threadable class")
