"""cthreads.linalg plugins (methods, ctors, array properties)."""

from .. import register_attr, register_call
from .attrs import LinalgPropPlugin
from .ctors import LinalgCtorPlugin
from .methods import LinalgMethodPlugin

register_call(LinalgMethodPlugin())
register_call(LinalgCtorPlugin())
register_attr(LinalgPropPlugin())
