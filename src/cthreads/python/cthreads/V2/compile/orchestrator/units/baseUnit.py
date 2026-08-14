from abc import ABC, abstractmethod
from dataclasses import dataclass

from .handle import Handle


@dataclass
class BaseUnit(ABC):

    handle: Handle

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def emit(self) -> None:
        """Emits the unit"""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validates the unit"""
        pass
