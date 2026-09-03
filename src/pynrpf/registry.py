"""Registry mapping model identifiers to their plugin implementations.

Inputs:  a model identifier string, such as ``m7_dtr`` or ``m8_xgb``.
Outputs: the plugin instance that implements inference (and, where supported,
         training) for that model.
Key steps: instantiate each known plugin once at import, then look it up by name,
         raising with the list of supported names when the lookup fails.
"""

from __future__ import annotations

from .plugins import M7DTRPlugin, M8XGBPlugin
from .plugins.base import BaseModelPlugin

_REGISTRY: dict[str, BaseModelPlugin] = {
    "m7_dtr": M7DTRPlugin(),
    "m8_xgb": M8XGBPlugin(),
}


def list_models() -> list[str]:
    """Return the registered model identifiers, sorted alphabetically."""
    return sorted(_REGISTRY.keys())


def get_model(model_name: str) -> BaseModelPlugin:
    """Return the plugin registered under a model identifier.

    Args:
        model_name: Model id, such as ``m7_dtr`` or ``m8_xgb``. Surrounding
            whitespace is ignored.

    Returns:
        The plugin instance implementing that model.

    Raises:
        KeyError: If the id is not registered. The message lists the supported ids.
    """
    name = str(model_name).strip()
    if name not in _REGISTRY:
        supported = ", ".join(list_models())
        raise KeyError(f"Unsupported model '{name}'. Supported models: {supported}")
    return _REGISTRY[name]
