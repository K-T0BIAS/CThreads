import os
from pathlib import Path

from ...frontend.Registry import REGISTRY
from ...types import PyType, PyThreadable, PyList, PyDict, PyTBuffer, PyShared


def add_include(bucket: list[str], seen: set[str], text: str) -> None:
    """Append unique `#include ...` lines (text may contain several)."""
    for line in text.splitlines(keepends=True):
        if line and line not in seen:
            seen.add(line)
            bucket.append(line)


def include_for(py_type: PyType, this_file: Path | str) -> str:
    """
    `#include` lines needed to use this type in the generated file `this_file`.

    `this_file` is the .hpp/.cpp being written, not the Python source.
    Nested list/dict/TBuffer inners are resolved so `list[Particle]` still
    picks up Particle.hpp (PyThreadable.build_include is empty).
    """
    if isinstance(py_type, PyThreadable):
        unit = REGISTRY.threadable_units.get(py_type.name)
        if unit is None:
            raise RuntimeError(
                f"Threadable {py_type.name!r} has no unit; "
                "call CompileSession.compile() before translation"
            )
        hpp = unit.hpp_path.resolve()
        dest = Path(this_file).resolve()
        if hpp == dest:
            return ""
        rel = os.path.relpath(hpp, dest.parent).replace("\\", "/")
        return f'#include "{rel}"\n'

    if isinstance(py_type, PyList):
        return PyType.build_include(py_type) + include_for(py_type.inner_type, this_file)
    if isinstance(py_type, PyDict):
        return (
            PyType.build_include(py_type)
            + include_for(py_type.key_type, this_file)
            + include_for(py_type.value_type, this_file)
        )
    if isinstance(py_type, PyTBuffer):
        return PyType.build_include(py_type) + include_for(py_type.inner_type, this_file)

    if isinstance(py_type, PyShared):
        return PyType.build_include(py_type) + include_for(py_type.inner_type, this_file)

    return py_type.build_include()
