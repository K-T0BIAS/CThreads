"""cthreads.sync plugins. Importing this package registers them."""

from .. import register_call
from .Sync import SyncMethodPlugin, SyncStatePlugin, TBufferMethodPlugin

register_call(SyncMethodPlugin())
register_call(TBufferMethodPlugin())
register_call(SyncStatePlugin())
