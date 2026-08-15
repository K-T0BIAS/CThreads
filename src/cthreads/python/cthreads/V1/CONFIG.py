"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
        # Idempotent: re-decoration / prepare(force=True) overwrites.
        self.threadables[cls.__name__] = cls

    def register_thread(self, fn) -> None:
        self.threads[fn.__qualname__] = fn

    def clear(self) -> None:
        self.threadables.clear()
        self.threads.clear()


REGISTRY = _Registry()
