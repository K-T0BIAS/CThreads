"""
Drain REGISTRY -> generate C++ -> fill STORE.

Call explicitly (or via prepare / api.thread). Order:
  1. all @Threadable classes (struct + @Thread methods)
  2. remaining free @Thread functions
"""

import inspect
from pathlib import Path

from .CONFIG import REGISTRY, KERNELS, STORE
from .cache import load_cache, save_cache


def compile(force: bool = False) -> dict:
    """
    Compiles all registered @Threadable / @Thread units into C++.

    Args:
        force: if True, ignore src hashes and regenerate all units.

    Returns:
        dict with keys: root, cache, rewritten (list of unit names written)
    """
    from .Thread.compile.compile import compile_free_thread
    from .Threadable.compile import compile_threadable

    KERNELS.clear()
    STORE.clear()
    claimed_methods: set[str] = set()
    rewritten: list[str] = []
    roots: set[Path] = set()

    # Infer project root early from any registered object.
    sample = None
    if REGISTRY.threadables:
        sample = next(iter(REGISTRY.threadables.values()))
    elif REGISTRY.threads:
        sample = next(iter(REGISTRY.threads.values()))
    if sample is None:
        raise RuntimeError("Nothing registered to compile")

    root = Path(inspect.getfile(sample)).resolve().parent
    cache = load_cache(root)

    for cls in list(REGISTRY.threadables.values()):
        methods = [
            fn
            for _, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
            if getattr(fn, "__threaded", False)
        ]
        changed = compile_threadable(cls, methods, force=force, cache=cache)
        if changed:
            rewritten.append(cls.__name__)
        for fn in methods:
            claimed_methods.add(fn.__qualname__)
        roots.add(Path(inspect.getfile(cls)).resolve().parent)

    for qualname, fn in list(REGISTRY.threads.items()):
        if qualname in claimed_methods:
            continue
        changed = compile_free_thread(fn, force=force, cache=cache)
        if changed:
            rewritten.append(fn.__name__)
        roots.add(Path(inspect.getfile(fn)).resolve().parent)

    # Keep REGISTRY so prepare(force=True) / second compile() still sees units.
    cache["units"] = cache.get("units", {})
    save_cache(root, cache)
    return {"root": root, "cache": cache, "rewritten": rewritten}
