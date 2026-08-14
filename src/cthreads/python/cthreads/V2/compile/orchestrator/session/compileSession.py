from pathlib import Path
import inspect
from typing import Any, get_type_hints

from ..units import ThreadUnit, ThreadableUnit, Handle
from ....frontend.Registry import REGISTRY
from ....types import PyType, hint_to_pytype
from ....kernel_meta import KERNELS, write_tbuffer_runtime
from ....cache import ensure_gitignore


class CompileSession:
    """
    Stateless compile pass: drain REGISTRY classes/functions into units
    stored back on REGISTRY. Does not keep its own unit maps.
    """

    @staticmethod
    def compile() -> dict[str, Any]:
        # ensure fresh registry / units / dispatch tables
        REGISTRY.threadable_units.clear()
        REGISTRY.thread_units.clear()
        KERNELS.clear()

        if not REGISTRY.threadables and not REGISTRY.threads:
            raise RuntimeError("Nothing registered to compile")

        if REGISTRY.threadables:
            sample: Any = next(iter(REGISTRY.threadables.values()))
        else:
            sample = next(iter(REGISTRY.threads.values()))
        root = Path(inspect.getfile(sample)).resolve().parent
        ensure_gitignore(root)

        claimed_methods: set[str] = set()

        for cls in list(REGISTRY.threadables.values()):
            src_file = Path(inspect.getfile(cls)).resolve()
            name = cls.__name__
            out_dir = src_file.parent / "__Threadable__"
            handle = Handle(name=name, path=str(src_file), target=cls)
            fields = {
                field_name: hint_to_pytype(hint)
                for field_name, hint in get_type_hints(
                    cls, localns={name: cls}
                ).items()
            }
            unit = ThreadableUnit(
                handle=handle,
                fields=fields,
                hpp_path=out_dir / f"{name}.hpp",
                cpp_path=out_dir / f"{name}.cpp",
            )
            for _, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not getattr(fn, "__threaded", False):
                    continue
                hints = get_type_hints(fn, localns={name: cls})
                params: list[tuple[str, PyType]] = []
                for pname in inspect.signature(fn).parameters:
                    if pname == "self":
                        continue
                    if pname not in hints:
                        raise TypeError(
                            f"Thread function {fn.__qualname__}: "
                            f"parameter {pname!r} needs a type annotation"
                        )
                    params.append((pname, hint_to_pytype(hints[pname])))
                ret_hint = hints.get("return")
                return_type = (
                    None
                    if ret_hint in (None, type(None))
                    else hint_to_pytype(ret_hint)
                )
                method_unit = ThreadUnit(
                    handle=Handle(
                        name=fn.__qualname__,
                        path=str(Path(inspect.getfile(fn)).resolve()),
                        target=fn,
                    ),
                    owner=unit,
                    params=params,
                    return_type=return_type,
                    hpp_path=None,
                    cpp_path=None,
                )
                unit.methods.append(method_unit)
                REGISTRY.thread_units[fn.__qualname__] = method_unit
                claimed_methods.add(fn.__qualname__)
            REGISTRY.threadable_units[name] = unit

        for qualname, fn in list(REGISTRY.threads.items()):
            if qualname in claimed_methods:
                continue
            src_file = Path(inspect.getfile(fn)).resolve()
            out_dir = src_file.parent / "__Thread__"
            hints = get_type_hints(fn)
            params: list[tuple[str, PyType]] = []
            for pname in inspect.signature(fn).parameters:
                if pname not in hints:
                    raise TypeError(
                        f"Thread function {qualname}: "
                        f"parameter {pname!r} needs a type annotation"
                    )
                params.append((pname, hint_to_pytype(hints[pname])))
            ret_hint = hints.get("return")
            return_type = (
                None
                if ret_hint in (None, type(None))
                else hint_to_pytype(ret_hint)
            )
            thread_unit = ThreadUnit(
                handle=Handle(
                    name=qualname,
                    path=str(src_file),
                    target=fn,
                ),
                owner=None,
                params=params,
                return_type=return_type,
                hpp_path=out_dir / f"{fn.__name__}.hpp",
                cpp_path=out_dir / f"{fn.__name__}.cpp",
            )
            REGISTRY.thread_units[qualname] = thread_unit

        for unit in REGISTRY.threadable_units.values():
            unit.emit()
        for unit in REGISTRY.thread_units.values():
            unit.emit()

        write_tbuffer_runtime(root)
        return {"root": root}
