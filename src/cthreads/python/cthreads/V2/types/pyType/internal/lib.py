from typing import Any

def is_internal_cthreads_type(item: Any) -> bool:
    """
    Check if the type is an internal CThreads type.
    """
    return isinstance(item, type) and getattr(item, "__cthreads_internal__", False)
