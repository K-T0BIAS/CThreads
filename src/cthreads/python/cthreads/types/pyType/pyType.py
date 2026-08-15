from dataclasses import dataclass


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
