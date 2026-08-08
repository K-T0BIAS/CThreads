VERSION: str = "0.1.0"

# Compiled outputs: name -> path to generated .hpp
STORE: dict[str, str] = {}

# Set by api.build.build() — shared library path for thread dispatch
BINARY_PATH: str | None = None

# Filled by compile(): symbol -> KernelMeta (dispatch schema)
KERNELS: dict = {}


class _Registry:
    """
    Intermediate definitions registered by @Threadable / @Thread.
    Drained by api.compile() into STORE + generated files.
    """

    def __init__(self) -> None:
        self.threadables: dict[str, type] = {}
        # qualname -> function (free or method)
        self.threads: dict[str, object] = {}

    def register_threadable(self, cls: type) -> None:
        name = cls.__name__
        if name in self.threadables:
            raise TypeError(f"Threadable {name!r} is already registered")
        self.threadables[name] = cls

    def register_thread(self, fn) -> None:
        key = fn.__qualname__
        if key in self.threads:
            raise TypeError(f"Thread {key!r} is already registered")
        self.threads[key] = fn

    def clear(self) -> None:
        self.threadables.clear()
        self.threads.clear()


REGISTRY = _Registry()
