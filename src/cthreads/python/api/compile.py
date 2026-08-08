"""
Drain REGISTRY → generate C++ → fill STORE.

Call explicitly (later: from the thread dispatch binding). Order:
  1. all @Threadable classes (struct + @Thread methods)
  2. remaining free @Thread functions
"""

import inspect

from .CONFIG import REGISTRY


def compile() -> None:
    """
    Compiles all @Threadable classes and free @Thread functions that are registered in the _Registry into c++ code

    Returns:
        None

    NOTE: this function wil write its output into the __Threadable__ and __Thread__ directories whare code was generated.
    """
    from .Thread.compile.compile import compile_free_thread # compile free @Thread functions
    from .Threadable.compile import compile_threadable # compile @Threadable classes

    claimed_methods: set[str] = set() # set of methods that have been claimed by a @Threadable class

    for cls in list(REGISTRY.threadables.values()): # iterate over all @Threadable classes
        methods = [ # get all methods that are @Threaded (this is done sepperate for this ptr access validity)
            fn
            for _, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
            if getattr(fn, "__threaded", False)
        ]
        compile_threadable(cls, methods)
        for fn in methods:
            claimed_methods.add(fn.__qualname__)
    # iterate over all free @Thread functions
    for qualname, fn in list(REGISTRY.threads.items()):
        if qualname in claimed_methods:
            continue
        compile_free_thread(fn)

    REGISTRY.clear() # clear the registry to prevent duplicate compilation
