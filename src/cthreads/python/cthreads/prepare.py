"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

High-level prepare + thread entry for V2.
"""

from pathlib import Path
from typing import Any

from .build import build
from .compiler.orchestrator import CompileSession
from .job import Job, wrap_job


def compile(force: bool = False) -> dict:
    """
    Drain the V2 registry into units, emit C++, write tbuffer runtime.

    Units that are unchanged (if the src_hash matches the previous one) skip translate/emit and only.
    Refreshes the kernel meta unless `force` is True.
    """
    return CompileSession.compile(force=force)


def prepare(force: bool = False) -> Path:
    """
    Runs Codegen and builds the native code (native code is the compiled dll file)

    Does NOT unload a loaded kernel library (cthreads.unload_kernels()).

    #### Warnings/Notes:
    - On Windows a force-relink over an already-loaded DLL will fail. Call `unload_kernels()` first. (This is only relevant if u compile multiple times in the same process)
    """
    info = compile(force=force)
    return build(project_root=info["root"], force=force)


def thread(fn, *args: Any, force: bool = False, **kwargs: Any) -> Job:
    """
    Puts the function `fn` into a C++ thread and returns a `Job` object for state management of this thread.
    The Job object is awaitable and will return the result of the function when it is done.

    #### Arguments:
    - `fn`: The function to put into a C++ thread.
    - `*args`: Arguments to pass to the function.
    - `force`: If True, force a retranslation of all the @Threads and @Threadables and a recompile of the generated C++ code.
    - `**kwargs`: Keyword arguments to pass to the function.

    #### Returns:
    - A `Job` object that can be awaited to get the result of the function.

    #### Example:
    ```python
    import cthreads
    from cthreads import Thread

    @Thread
    def my_function(name: str) -> str:
        res: str = "Hello, "+name+"!"
        return res

    job = cthreads.thread(my_function, "World")
    job.join()
    result = job.result()
    print(result)
    ```
    """
    from . import _ext_api

    loaded = _ext_api.kernel_path()
    if force:
        if loaded:
            raise RuntimeError(
                "cthreads.thread(force=True): kernels are still loaded. "
                "Call unload_kernels() before force-rebuild, then "
                "thread(..., force=True) or prepare(force=True)+load_kernels."
            )
        binary = prepare(force=True)
        _ext_api.load_kernels(str(binary))
    elif not loaded:
        binary = prepare(force=False)
        _ext_api.load_kernels(str(binary))

    # Wrap the threaded function in a Job object and return it.
    return wrap_job(_ext_api.thread(fn, *args, **kwargs))
