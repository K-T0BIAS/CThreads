"""stdlib plugins (math → cmath, len / list / dict, …)."""

from .. import register_attr, register_call
from .Containers import ContainerMethodPlugin, LenPlugin
from .Math import MathCallPlugin, MathConstPlugin

register_call(LenPlugin())
register_call(ContainerMethodPlugin())
register_call(MathCallPlugin())
register_attr(MathConstPlugin())
