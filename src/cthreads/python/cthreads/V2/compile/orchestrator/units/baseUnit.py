from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .handle import Handle


@dataclass
class BaseUnit(ABC):

    handle: Handle

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def emit(self, *, force: bool = False, cache: dict[str, Any] | None = None) -> bool:
        """Emit the unit. Returns True if generated files were rewritten."""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validates the unit"""
        pass
