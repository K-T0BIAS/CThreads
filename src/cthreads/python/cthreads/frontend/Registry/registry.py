"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Registry for threadable and thread functions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...compiler.orchestrator.units import ThreadableUnit, ThreadUnit
    from ...gpu.compiler.orchestrator.gpu_unit import GpuUnit

class Registry:
    """
    Decorators register live classes / functions.
    CompileSession fills threadable_units / thread_units and does not keep its own maps.
    """

    VERSION: str = "0.1.0"  # used for code generation and binary version tracking

    def __init__(self) -> None:
        # class name -> class (@Threadable)
        self.threadables: dict[str, type] = {}
        # function qualname -> function (@Thread)
        self.threads: dict[str, object] = {}
        # filled by CompileSession.compile()
        self.threadable_units: dict[str, ThreadableUnit] = {}
        self.thread_units: dict[str, ThreadUnit] = {}

        # gpu stuff
        self.gpu_functions: dict[str, object] = {} # function qualname -> function (@Gpu)
        self.gpu_function_units: dict[str, GpuUnit] = {} # function qualname -> GpuUnit

    def register_threadable(self, cls: type) -> None:
        """Registers a threadable class for code generation"""
        self.threadables[cls.__name__] = cls

    def register_thread(self, fn: object) -> None:
        """Registers a thread function for code generation"""
        self.threads[fn.__qualname__] = fn

    def register_gpu_function(self, fn: object) -> None:
        """Registers a gpu function for code generation"""
        self.gpu_functions[fn.__qualname__] = fn

    def clear(self) -> None:
        """Clears the registry"""
        self.threadables.clear()
        self.threads.clear()
        self.threadable_units.clear()
        self.thread_units.clear()
        self.gpu_functions.clear()
        self.gpu_function_units.clear()
        from ...kernel_meta import KERNELS
        from ...gpu.gpu_kernel_meta import GPU_KERNELS

        KERNELS.clear()
        GPU_KERNELS.clear()


REGISTRY = Registry()
