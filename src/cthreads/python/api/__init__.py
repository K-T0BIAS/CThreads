"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Backward-compatible shim. Prefer `import cthreads`.
"""

from cthreads import *  # noqa: F403
from cthreads import __all__  # noqa: F401
