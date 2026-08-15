"""cthreads.math plugins. Importing this package registers them."""

from .. import register_call
from .Math import CthreadsMathCallPlugin

register_call(CthreadsMathCallPlugin())
