from __future__ import annotations

from .plugins import M7DTRPlugin, M8XGBPlugin
from .plugins.base import BaseModelPlugin

_REGISTRY: dict[str, BaseModelPlugin] = {
    "m7_dtr": M7DTRPlugin(),
    "m8_xgb": M8XGBPlugin(),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model(model_name: str) -> BaseModelPlugin:
    name = str(model_name).strip()
    if name not in _REGISTRY:
        supported = ", ".join(list_models())
        raise KeyError(f"Unsupported model '{name}'. Supported models: {supported}")
    return _REGISTRY[name]
