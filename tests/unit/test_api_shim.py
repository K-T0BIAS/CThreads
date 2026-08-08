"""api package remains a thin re-export shim."""

import api


def test_api_reexports_cthreads():
    assert hasattr(api, "Thread")
    assert hasattr(api, "Threadable")
    assert hasattr(api, "compile")
