from ..pyType import PyType

class PyCThreadsInternalType(PyType):
    """Maps a marked cthreads.sync / cthreads.linalg type to its C++ class + header."""

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
