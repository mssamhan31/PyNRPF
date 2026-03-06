"""PyNRPF implementation package API."""

from .api import run_inference, train_m8_xgb
from .artifacts import load_artifact_bundle, save_artifact_bundle
from .config import load_config
from .registry import list_models
from .scaffold import (
    build_pipeline_config,
    generate_model_scaffold,
    generate_pipeline_config,
)

__all__ = [
    "load_artifact_bundle",
    "load_config",
    "list_models",
    "run_inference",
    "save_artifact_bundle",
    "train_m8_xgb",
    "build_pipeline_config",
    "generate_pipeline_config",
    "generate_model_scaffold",
]

__version__ = "0.3.0"
