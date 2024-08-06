from .dispatch import felinewhiskercli
from .init import _add_init_subcommand

_DECORATORS = [
    _add_init_subcommand,
]

cli = felinewhiskercli
for deco in _DECORATORS:
    cli = deco(cli)
