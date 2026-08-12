"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
from typing import Any, get_args, get_origin

from .Threadable.lib import is_internal_cthreads_type


@dataclass
class PyType:
    """
    Default type for python type to c++ type mapping

    Attributes:
        name: the name of the python type
        cpp_name: the name of the c++ type
        description: the description of the c++ type
        cpp_include: the include statement for the c++ type
        needs_include: whether the c++ type needs an include statement

    Methods:
        build_include: builds the include statement for the c++ type
        to_cpp: converts a python type to a c++ type (translates a python style decl to the c++ equivalent)
    """

    name: str
    cpp_name: str
    description: str
    cpp_include: str
    needs_include: bool

    def build_include(self) -> str:
        """Builds the include statement for the c++ type"""
        if self.needs_include:
            return f"#include <{self.cpp_include}>\n"
        return ""

    def to_cpp(self, var_name: str, default_value: str | None = None) -> tuple[str, str]:
        """
        Converts a python type to a c++ type (translates a python style decl to the c++ equivalent)

        Args:
            var_name: the name of the variable
            default_value: the default value of the variable

        Returns:
            tuple[str, str]: the c++ declaration and the include statement

        Example:
            PyInt().to_cpp("x") -> "int x;", ""
            PyInt().to_cpp("x", "10") -> "int x = 10;", ""
            PyString().to_cpp("x") -> "std::string x;", "string"
            PyString().to_cpp("x", "10") -> "std::string x = 10;", "string"
            PyBool().to_cpp("x") -> "bool x;", ""
            PyBool().to_cpp("x", "True") -> "bool x = True;", ""
            PyList(PyInt()).to_cpp("x") -> "std::vector<int> x;", "vector"
            PyList(PyInt()).to_cpp("x", "[1, 2, 3]") -> "std::vector<int> x = [1, 2, 3];", "vector"
        """
        if default_value is None:
            return f"{self.cpp_name} {var_name};", self.build_include()
        return f"{self.cpp_name} {var_name} = {default_value};", self.build_include()


class PyInt(PyType):
    """conversion type for python int to c++ int"""
    def __init__(self) -> None:
        super().__init__(
            name="int",
            cpp_name="int",
            description="int",
            cpp_include="",
            needs_include=False,
        )


class PyFloat(PyType):
    """conversion type for python float to c++ double"""
    def __init__(self) -> None:
        super().__init__(
            name="float",
            cpp_name="double",
            description="double",
            cpp_include="",
            needs_include=False,
        )


class PyString(PyType):
    """conversion type for python string to c++ std::string"""
    def __init__(self) -> None:
        super().__init__(
            name="str",
            cpp_name="std::string",
            description="std::string",
            cpp_include="string",
            needs_include=True,
        )


class PyBool(PyType):
    """conversion type for python bool to c++ bool"""
    def __init__(self) -> None:
        super().__init__(
            name="bool",
            cpp_name="bool",
            description="bool",
            cpp_include="",
            needs_include=False,
        )


class PyList(PyType):
    """
    conversion type for python list to c++ std::vector
    
    Attributes:
        inner_type: the type of the elements in the list

    Methods:
        build_include: builds the include statement for the c++ type
        to_cpp: converts a python list to a c++ std::vector

    Example:
        PyList(PyInt()).to_cpp("x") -> "std::vector<int> x;", "vector"
    """
    inner_type: PyType

    def __init__(self, inner_type: PyType) -> None:
        self.inner_type = inner_type
        super().__init__(
            name="list",
            cpp_name=f"std::vector<{self.inner_type.cpp_name}>",
            description=f"std::vector<{self.inner_type.description}>",
            cpp_include="vector",
            needs_include=True,
        )

    def build_include(self) -> str:
        return super().build_include() + self.inner_type.build_include()


class PyDict(PyType):
    """
    conversion type for python dict to c++ std::unordered_map

    Attributes:
        key_type: the type of the keys in the dict
        value_type: the type of the values in the dict

    Methods:
        build_include: builds the include statement for the c++ type
        to_cpp: converts a python dict to a c++ std::unordered_map

    Example:
        PyDict(PyString(), PyInt()).to_cpp("x") -> "std::unordered_map<std::string, int> x;", "unordered_map"
        PyDict(PyString(), PyInt()).to_cpp("x", "{'a': 1, 'b': 2}") -> "std::unordered_map<std::string, int> x = {'a': 1, 'b': 2};", "unordered_map"
    """
    key_type: PyType
    value_type: PyType

    def __init__(self, key_type: PyType, value_type: PyType) -> None:
        self.key_type = key_type
        self.value_type = value_type
        super().__init__(
            name="dict",
            cpp_name=f"std::unordered_map<{key_type.cpp_name}, {value_type.cpp_name}>",
            description=f"std::unordered_map<{key_type.description}, {value_type.description}>",
            cpp_include="unordered_map",
            needs_include=True,
        )

    def build_include(self) -> str:
        return (
            super().build_include()
            + self.key_type.build_include()
            + self.value_type.build_include()
        )


class PyThreadable(PyType):
    """
    conversion type for a python @Threadable class to a c++ struct
    """
    def __init__(self, name: str, location: str) -> None:
        self.location = location
        super().__init__(
            name=name,
            cpp_name=name,
            description="Threadable",
            cpp_include=location,
            needs_include=True,
        )

    def build_include(self) -> str:
        return f'#include "{self.cpp_include}"\n'


class TBuffer:
    """
    Annotation helper for triple-buffer kernel params.

    Use ``TBuffer[Particle]`` in @Thread signatures; codegen emits
    ``cthreads::sync::tripple_buffer<Particle>``.
    """

    __cthreads_tbuffer__ = True

    def __class_getitem__(cls, inner: Any) -> type:
        name = getattr(inner, "__name__", repr(inner))
        stub = type(f"TBuffer[{name}]", (), {})
        stub.__cthreads_tbuffer__ = True
        stub.__cthreads_tbuffer_inner__ = inner
        return stub


class PyTBuffer(PyType):
    """Maps ``TBuffer[inner]`` to ``cthreads::sync::tripple_buffer<inner>``."""

    inner_type: PyType

    def __init__(self, inner_type: PyType) -> None:
        self.inner_type = inner_type
        cpp_inner = inner_type.cpp_name
        super().__init__(
            name="TBuffer",
            cpp_name=f"cthreads::sync::tripple_buffer<{cpp_inner}>",
            description=f"tripple_buffer<{inner_type.description}>",
            cpp_include="sync/t_buffer.hpp",
            needs_include=True,
        )

    def build_include(self) -> str:
        return super().build_include() + self.inner_type.build_include()


TBUFFER_INTERNAL_NAMES: frozenset[str] = frozenset(
    {
        "TBufferF64",
        "TBufferI64",
        "TBufferBool",
        "TBufferStr",
        "TBufferListF64",
        "TBufferDictStrF64",
        "TBufferObj",
    }
)


def is_tbuffer_pytype(py_type: PyType) -> bool:
    """True for ``TBuffer[...]`` and fixed ``cthreads.sync.TBuffer*`` types."""
    return isinstance(py_type, PyTBuffer) or (
        isinstance(py_type, PyCthreadsInternal)
        and py_type.name in TBUFFER_INTERNAL_NAMES
    )


# Native sync types shipped in the pybind module (not codegen'd as Threadables).
# keyed by Python class __name__ as exposed on cthreads.sync
CTHREADS_INTERNAL_TYPES: dict[str, dict[str, str]] = {
    "Lock": {
        "cpp_name": "cthreads::sync::Lock",
        "cpp_include": "sync/pyLock.hpp",
    },
    "Event": {
        "cpp_name": "cthreads::sync::Event",
        "cpp_include": "sync/pyEvent.hpp",
    },
    "RWLock": {
        "cpp_name": "cthreads::sync::RWLock",
        "cpp_include": "sync/pyRWLock.hpp",
    },
    # Fixed-capacity triple buffers (cthreads.sync.TBuffer*).
    "TBufferF64": {
        "cpp_name": "cthreads::sync::tripple_buffer<double>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferI64": {
        "cpp_name": "cthreads::sync::tripple_buffer<int>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferBool": {
        "cpp_name": "cthreads::sync::tripple_buffer<bool>",
        "cpp_include": "sync/t_buffer.hpp",
    },
    "TBufferStr": {
        "cpp_name": "cthreads::sync::tripple_buffer<std::string>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("string",),
    },
    "TBufferListF64": {
        "cpp_name": "cthreads::sync::tripple_buffer<std::vector<double>>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("vector",),
    },
    "TBufferDictStrF64": {
        "cpp_name": (
            "cthreads::sync::tripple_buffer<std::unordered_map<std::string, double>>"
        ),
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("string", "unordered_map"),
    },
    "TBufferObj": {
        "cpp_name": "cthreads::sync::tripple_buffer<py::object>",
        "cpp_include": "sync/t_buffer.hpp",
        "extra_includes": ("pybind11/pybind11.h",),
    },
}


class PyCthreadsInternal(PyType):
    """Maps a marked cthreads.sync type to its existing C++ class + header."""

    extra_includes: tuple[str, ...]

    def __init__(
        self,
        py_name: str,
        cpp_name: str,
        cpp_include: str,
        extra_includes: tuple[str, ...] = (),
    ) -> None:
        self.extra_includes = extra_includes
        super().__init__(
            name=py_name,
            cpp_name=cpp_name,
            description=cpp_name,
            cpp_include=cpp_include,
            needs_include=True,
        )

    def build_include(self) -> str:
        out = f'#include "{self.cpp_include}"\n'
        for inc in self.extra_includes:
            out += f"#include <{inc}>\n"
        return out


def hint_to_pytype(hint: Any) -> PyType:
    """
    Helper to build PyType from a python type hint

    Args:
        hint: the python type hint

    Returns:
        PyType: the PyType object

    Example:
        hint_to_pytype(int) -> PyInt()
        hint_to_pytype(list[int]) -> PyList(PyInt())
        hint_to_pytype(dict[str, int]) -> PyDict(PyString(), PyInt())
        hint_to_pytype(Threadable) -> PyThreadable("Threadable", "Threadable.hpp")
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

    # cthreads.sync.* marked with __cthreads_internal__ — link headers, don't generate
    if is_internal_cthreads_type(hint):
        entry = CTHREADS_INTERNAL_TYPES.get(hint.__name__)
        if entry is None:
            raise TypeError(
                f"Internal cthreads type {hint.__name__!r} is not in CTHREADS_INTERNAL_TYPES"
            )
        return PyCthreadsInternal(
            py_name=hint.__name__,
            cpp_name=entry["cpp_name"],
            cpp_include=entry["cpp_include"],
            extra_includes=tuple(entry.get("extra_includes", ())),
        )

    # if the hint points to a @Threadable class it must be resolved to ensure correct includes
    if isinstance(hint, type) and getattr(hint, "__threadable", False):
        from .CONFIG import STORE # get the store to lookup this class
        # get the location of the Threadable class (default to the name.hpp in the __Threadable__ directory (local))
        location = STORE.get(hint.__name__, f"__Threadable__/{hint.__name__}.hpp")
        return PyThreadable(hint.__name__, location) # return the PyThreadable object

    raise TypeError(f"Cannot map type {hint!r} to a PyType") # raise an error if the type is not supported
