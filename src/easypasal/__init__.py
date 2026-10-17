"""EasyPasal: Evaluated, cited answers on Indonesian financial regulation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.config import EasyPasalConfig

__version__ = "0.1.0"
__all__ = ["EasyPasalConfig", "__version__"]


def __getattr__(name: str) -> object:
    if name == "EasyPasalConfig":
        from easypasal.config import EasyPasalConfig

        return EasyPasalConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
