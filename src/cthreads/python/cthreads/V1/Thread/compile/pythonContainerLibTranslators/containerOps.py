from collections.abc import Callable
from typing import NamedTuple

# already-translated C++ exprs
Emit = Callable[[str, list[str]], str]  # (receiver_cpp, arg_cpps) -> expr

class ContainerOp(NamedTuple):
    emit: Emit
    min_arity: int
    max_arity: int
    cpp_include: str | None = None  # usually None; vector/map already on the type

LIST_METHODS: dict[str, ContainerOp] = {
    "append": ContainerOp(
        emit=lambda recv, args: f"({recv}).push_back({args[0]})",
        min_arity=1,
        max_arity=1,
    ),
    "clear": ContainerOp(
        emit=lambda recv, args: f"({recv}).clear()",
        min_arity=0,
        max_arity=0,
    ),
    "pop": ContainerOp(
        emit=lambda recv, args: (
            f"([&]() {{"
            f"   auto v = ({recv}).back();"
            f"   ({recv}).pop_back();"
            f"   return v;"
            f"}}())"
            if not args
            else f"([&]() {{"
            f"   auto& c = ({recv});"
            f"   auto it = c.begin() + ({args[0]});"
            f"   auto v = *it;"
            f"   c.erase(it);"
            f"   return v;"
            f"}}())"
        ),
        min_arity=0,
        max_arity=1,
    ),
    "insert": ContainerOp(
        emit=lambda recv, args: (
            f"({recv}).insert(({recv}).begin() + ({args[0]}), {args[1]})"
        ),
        min_arity=2,
        max_arity=2,
    ),
    "extend": ContainerOp(
        emit=lambda recv, args: (
            f"({recv}).insert(({recv}).end(), ({args[0]}).begin(), ({args[0]}).end())"
        ),
        min_arity=1,
        max_arity=1,
    ),
}

DICT_METHODS: dict[str, ContainerOp] = {
    # get(k) -> None is unsupported (no Optional/None in Thread); require default.
    "get": ContainerOp(
        emit=lambda recv, args: (
            f"([&]() {{ "
            f"    auto it = ({recv}).find({args[0]}); "
            f"    return it != ({recv}).end() ? it->second : ({args[1]}); "
            f"}}())"
        ),
        min_arity=2,
        max_arity=2,
    ),
    "clear": ContainerOp(
        emit=lambda recv, args: f"({recv}).clear()",
        min_arity=0,
        max_arity=0,
    ),
    # pop(k) raises KeyError - unsupported until exceptions exist; require default.
    "pop": ContainerOp(
        emit=lambda recv, args: (
            f"([&]() {{ "
            f"    auto it = ({recv}).find({args[0]}); "
            f"    if (it == ({recv}).end()) return ({args[1]}); "
            f"    auto v = it->second; "
            f"    ({recv}).erase(it); "
            f"    return v; "
            f"}}())"
        ),
        min_arity=2,
        max_arity=2,
    ),
}

