"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ..Threadable.lib import is_threadable
from ..Registry import REGISTRY

def Thread(fn):
    """
    Decorator to mark a function as threadable in the cthreads api.

    #### Rules

    - All argumentts must be type hinted with a valid cthreadable type
    - The return type must be type hinted with a valid cthreadable type (or -> None)
    - No *args or **kwargs are allowed
    - Local variables must be `annotated` with a valid cthreadable type (x: int = 0)
    - The funtion must not use non-@Thread decorated functions except for `python math` functions and those from `cthreads.math`, `cthreads.sync`
    
    #### Properties
    - The function may become a method of a @Threadable class if it is decorated with @Threadable (Non-@Threadable classses can not use @Thread functions)
    - The function functions works like a normal ython function if called normally
    - The function can be called with `cthreads.thread(...)` to create a job

    #### Example

    ----
    >>>> from cthreads.Thread import Thread
    >>>> @Thread
    ... def my_function(x: int) -> int:
    ...     return x + 1
    >>>>
    >>>> # async awaitable job
    >>>> job = cthreads.thread(my_function, 1)
    >>>> result = await job
    >>>> print(result)
    ... 2
    >>>>
    >>>> # sync job
    >>>> job.start()
    >>>> result = job.result()
    >>>> print(result)
    ... 2
    
    ----

    #### Information for developers

    - the decorator wrapped functions that are not part of a @Threadable class are considered free functions in the c++ backend
    - upon calling `cthreads.thread(...)` all decorated functions are translated into c++ code and compiled
    - for each function the translation unit writes a `hpp` and `cpp` file in the `__Thread__` subfolder nextto the file that the python function is defined in
    the fn gets a `__threaded` attribute set to True and a `__thread_version` attribute set to the current version of the cthreads api

    ```
    my_module/
    |-- __Thread__/
    |       |-- my_function.hpp
    |       |-- my_function.cpp
    |       |-- ...
    |-- my_module.py # includes @Thread def my_function()
    |-- ...
    ```

    """
    fn.__threaded = True
    fn.__thread_version = REGISTRY.VERSION

    from typing import get_type_hints

    hints = get_type_hints(fn) # get the type hints of the function (arguments and return type)
    for name, hint in hints.items():
        # skip the return type if it is None or type(None)
        if name == "return" and hint in (None, type(None)):
            continue
        try:
            # check if the type is allowed in the cthreads api
            is_threadable(hint)
        except TypeError as e:
            # raise an error if the type is not allowed in the cthreads api
            raise TypeError(
                f"Thread function {fn.__name__} has invalid type for {name!r}: {hint}"
            ) from e

    REGISTRY.register_thread(fn) # register the function in the cthreads api
    return fn
