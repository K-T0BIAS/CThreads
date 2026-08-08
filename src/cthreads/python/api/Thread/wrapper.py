from ..CONFIG import VERSION, REGISTRY
from ..Threadable.lib import is_threadable


def Thread(fn):
    fn.__threaded = True
    fn.__thread_version = VERSION

    from typing import get_type_hints

    hints = get_type_hints(fn)
    for name, hint in hints.items():
        if name == "return" and hint in (None, type(None)):
            continue
        try:
            is_threadable(hint)
        except TypeError as e:
            raise TypeError(
                f"Thread function {fn.__name__} has invalid type for {name!r}: {hint}"
            ) from e

    REGISTRY.register_thread(fn)
    return fn
