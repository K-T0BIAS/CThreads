from typing import Any


class Cpp:
    """C++ text helpers used by Syntax (not node translators)."""

    CMATH = "#include <cmath>\n"
    NUMBERS = "#include <numbers>\n"

    @staticmethod
    def literal(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if value is None:
            raise TypeError("None is not a supported Thread literal")
        return str(value)
