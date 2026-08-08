"""
Backward-compatible shim. Prefer `import cthreads`.
"""

from cthreads import *  # noqa: F403
from cthreads import __all__  # noqa: F401
