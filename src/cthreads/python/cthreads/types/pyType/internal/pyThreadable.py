from ..pyType import PyType

class PyThreadable(PyType):
    """
    conversion type for a python @Threadable class to a c++ struct.

    Identity only (the class name). The generated header path lives on
    the ThreadableUnit in the registry and is resolved at emit time.
    """
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            cpp_name=name,
            description="Threadable",
            cpp_include="",
            needs_include=False,
        )
