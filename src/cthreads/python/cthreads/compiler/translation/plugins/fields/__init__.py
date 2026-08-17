"""Field / self attribute fallback. Register last."""

from .. import register_attr, register_call
from .Attr import FieldAttrPlugin, ThreadableMethodPlugin

register_attr(FieldAttrPlugin())
register_call(ThreadableMethodPlugin())
