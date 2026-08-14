from typing import Any, get_origin, get_args

from .pyType import PyType
from .internal import is_internal_cthreads_type, PyTBuffer, PyCThreadsInternalType, PyThreadable, CTHREADS_INTERNAL_TYPES
from .py import PyInt, PyFloat, PyString, PyBool, PyList, PyDict

def hint_to_pytype(hint: Any) -> PyType:
    """
    Helper to build PyType from a python type hint

    #### Args:
        hint: the python type hint

    #### Returns:
        PyType: the PyType object

    #### Example:
        hint_to_pytype(int) -> PyInt()
        hint_to_pytype(list[int]) -> PyList(PyInt())
        hint_to_pytype(dict[str, int]) -> PyDict(PyString(), PyInt())
        hint_to_pytype(Threadable) -> PyThreadable("Threadable")
    """

    origin = get_origin(hint)
    # chec for generic types
    if origin is list:
        (inner,) = get_args(hint) # get the inner type
        return PyList(hint_to_pytype(inner)) # return the PyList object
    if origin is dict:
        key, value = get_args(hint) # get the key and value types
        return PyDict(hint_to_pytype(key), hint_to_pytype(value)) # return the PyDict object
    if origin is set:
        raise TypeError("set is not supported for codegen yet") # raise an error if the type is not supported

    # check for primitive types
    primitives = {
        int: PyInt,
        float: PyFloat,
        str: PyString,
        bool: PyBool,
    }
    if hint in primitives:
        return primitives[hint]()

    # TBuffer[inner] — codegen uses tripple_buffer<inner_cpp> in the kernel.
    if getattr(hint, "__cthreads_tbuffer__", False):
        inner_hint = getattr(hint, "__cthreads_tbuffer_inner__", None)
        if inner_hint is None:
            args = get_args(hint)
            if args:
                inner_hint = args[0]
        if inner_hint is None:
            raise TypeError("TBuffer[...] requires an element type annotation")
        return PyTBuffer(hint_to_pytype(inner_hint))

    # cthreads.sync.* marked with __cthreads_internal__, link headers, don't generate
    if is_internal_cthreads_type(hint):
        entry = CTHREADS_INTERNAL_TYPES.get(hint.__name__)
        if entry is None:
            raise TypeError(
                f"Internal cthreads type {hint.__name__!r} is not in CTHREADS_INTERNAL_TYPES"
            )
        return PyCThreadsInternalType(
            py_name=hint.__name__,
            cpp_name=entry["cpp_name"],
            cpp_include=entry["cpp_include"],
            extra_includes=tuple(entry.get("extra_includes", ())),
        )

    # @Threadable class — name only; emit looks up the unit's hpp_path
    if isinstance(hint, type) and getattr(hint, "__threadable", False):
        return PyThreadable(hint.__name__)

    raise TypeError(f"Cannot map type {hint!r} to a PyType") # raise an error if the type is not supported
