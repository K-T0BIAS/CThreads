from dataclasses import dataclass

from ...types import PyType


@dataclass
class SignatureResult:
    """Output of translate_signature. Symbols stay on TranslationContext."""

    return_type: PyType | None
    func_name: str
    params_csv: str


@dataclass
class TranslationResult:
    """Emit contract for one translated function."""

    return_type: PyType | None
    func_name: str
    params_csv: str
    sig_includes: list[str]
    body: str
    body_includes: list[str]

    def _return_cpp(self) -> str:
        if self.return_type is None:
            return "void"
        return self.return_type.cpp_name

    def free_signature(self) -> str:
        return f"CTHREADS_API {self._return_cpp()} {self.func_name}({self.params_csv})"

    def method_decl(self) -> str:
        return f"    {self._return_cpp()} {self.func_name}({self.params_csv});"

    def method_def_signature(self, owner_name: str) -> str:
        return f"{self._return_cpp()} {owner_name}::{self.func_name}({self.params_csv})"
