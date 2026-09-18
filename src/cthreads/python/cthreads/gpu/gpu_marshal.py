"""
Marshal helpers for `gpu()` launch (ordered args + dispatch sizing).
"""

from typing import Any


def ordered_values_for_meta(meta: dict[str, Any], args: tuple[Any, ...]) -> list[Any]:
    """
    Build the ordered argument list for `launch_gpu_kernel`.

    #### Args:
    - meta: dict[str, Any] = kernel metadata (`params`, …)
    - args: tuple[Any, ...] = positional args from `gpu(fn, *args)`

    #### Returns
    - list[Any] = args in parameter order (same Python list objects for refs)

    #### Raises
    - TypeError = arity mismatch
    """
    params = meta.get("params")
    if not isinstance(params, list):
        raise TypeError("GPU meta is missing a params list")
    if len(args) != len(params):
        raise TypeError(
            f"GPU kernel {meta.get('symbol', '?')!r}: expected {len(params)} "
            f"args, got {len(args)}"
        )
    return list(args)


def infer_group_count_x(meta: dict[str, Any], ordered_values: list[Any]) -> int:
    """
    Choose `group_count_x` as ceil(n / local_size_x).

    Prefers a scalar parameter named `n`, else the longest list argument length.

    #### Args:
    - meta: dict[str, Any] = kernel metadata
    - ordered_values: list[Any] = launch args in param order

    #### Returns
    - int = workgroup count in X (>= 1)
    """
    local_size_x: int = int(meta.get("local_size_x") or 64)
    if local_size_x < 1:
        local_size_x = 64

    params = meta.get("params")
    if not isinstance(params, list):
        return 1

    n: int | None = None
    for i, param in enumerate(params):
        if not isinstance(param, dict):
            continue
        name = param.get("name")
        kind = param.get("kind")
        if name == "n" and kind == "int":
            n = int(ordered_values[i])
            break

    if n is None:
        best = 0
        for i, param in enumerate(params):
            if isinstance(param, dict) and param.get("kind") == "list":
                val = ordered_values[i]
                if isinstance(val, list):
                    best = max(best, len(val))
        n = best

    if n is None or n < 1:
        return 1
    return (int(n) + local_size_x - 1) // local_size_x
