from ..pyType import PyType

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
