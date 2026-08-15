"""Field / self attribute fallback. Register last."""

from .. import register_attr
from .Attr import FieldAttrPlugin

register_attr(FieldAttrPlugin())
