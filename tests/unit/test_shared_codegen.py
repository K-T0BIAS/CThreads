"""Codegen emission for Shared[T] kernels."""

from cthreads import Thread, Shared, compile


def test_compile_shared_kernel_emits_shared_host_trampoline(tmp_module):
    mod = tmp_module(
        """
        from cthreads import Thread, Shared

        @Thread
        def bump(head: Shared[list[int]], i: int) -> None:
            head[i] = head[i] + 1
        """,
        name="ct_shared_codegen",
    )
    info = compile(force=True)
    root = info["root"]
    cpp = (root / "__Thread__" / "bump.cpp").read_text(encoding="utf-8")
    assert "std::vector<int>& head" in cpp
    assert "bump__promote_a0_shared" in cpp
    assert "bump__demote_a0_shared" in cpp
    assert 'get<std::vector<int>>("head")' in cpp

    from cthreads.kernel_meta import KERNELS

    assert KERNELS["bump"].params[0].pass_as == "shared"
