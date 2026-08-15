from dataclasses import dataclass
from typing import Any


@dataclass
class Handle:
    """Live Python object plus the source file it was declared in."""

    name: str
    path: str
    target: Any
