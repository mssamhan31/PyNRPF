"""Model plugin implementations, exported for the registry to instantiate."""

from .m7_dtr import M7DTRPlugin
from .m8_xgb import M8XGBPlugin

__all__ = ["M7DTRPlugin", "M8XGBPlugin"]
