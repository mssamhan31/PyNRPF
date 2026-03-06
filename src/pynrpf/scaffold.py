from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import DEFAULT_CONFIG
from .training_config import DEFAULT_TRAINING_CONFIG

_MODEL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_model_id(model_id: str) -> str:
    model = str(model_id).strip()
    if not _MODEL_ID_PATTERN.match(model):
        raise ValueError(
            "model_id must match ^[a-z][a-z0-9_]*$ (for example: m9_custom)."
        )
    return model


def _to_class_name(model_id: str) -> str:
    return "".join(piece.capitalize() for piece in model_id.split("_"))


def _write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Set overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _plugin_template(model_id: str) -> str:
    class_name = _to_class_name(model_id)
    return f"""from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseModelPlugin


class {class_name}Plugin(BaseModelPlugin):
    name = "{model_id}"

    def run_inference(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        result = df.copy()
        net_col = columns["net_load"]
        result["pynrpf_interval_flag"] = False
        result["pynrpf_day_flag"] = False
        result["pynrpf_confidence"] = np.nan
        result["pynrpf_corrected_net_load"] = result[net_col]
        return result

    def train(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Implement training logic for {model_id} and return an artifact bundle dict."
        )
"""


def _test_template(model_id: str) -> str:
    class_name = _to_class_name(model_id)
    return f"""from pynrpf.plugins.{model_id} import {class_name}Plugin


def test_{model_id}_plugin_name() -> None:
    plugin = {class_name}Plugin()
    assert plugin.name == "{model_id}"
"""


def _ensure_plugins_init_wiring(root: Path, model_id: str) -> Path:
    class_name = f"{_to_class_name(model_id)}Plugin"
    path = root / "src" / "pynrpf" / "plugins" / "__init__.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot auto-wire plugin imports because file does not exist: {path}"
        )

    text = path.read_text(encoding="utf-8")
    import_matches = re.findall(r"^from \.([a-z0-9_]+) import ([A-Za-z0-9_]+)$", text, re.MULTILINE)
    import_map = {module: cls for module, cls in import_matches}
    import_map[model_id] = class_name

    ordered_modules = sorted(import_map.keys())
    lines = [f"from .{module} import {import_map[module]}" for module in ordered_modules]
    all_items = [import_map[module] for module in ordered_modules]
    lines.append("")
    lines.append(f"__all__ = {all_items!r}".replace("'", '"'))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ensure_registry_wiring(root: Path, model_id: str) -> Path:
    class_name = f"{_to_class_name(model_id)}Plugin"
    path = root / "src" / "pynrpf" / "registry.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot auto-wire model registry because file does not exist: {path}"
        )

    text = path.read_text(encoding="utf-8")

    import_match = re.search(r"^from \.plugins import ([^\n]+)$", text, re.MULTILINE)
    if not import_match:
        raise ValueError("Unable to locate '.plugins' import line in registry.py.")
    current_imports = [item.strip() for item in import_match.group(1).split(",") if item.strip()]
    if class_name not in current_imports:
        current_imports.append(class_name)
    current_imports = sorted(set(current_imports))
    new_import_line = f"from .plugins import {', '.join(current_imports)}"
    text = re.sub(r"^from \.plugins import [^\n]+$", new_import_line, text, flags=re.MULTILINE)

    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("_REGISTRY:") and line.strip().endswith("{"):
            start_idx = i
            continue
        if start_idx is not None and line.strip() == "}":
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise ValueError("Unable to locate _REGISTRY dictionary block in registry.py.")

    entry_map: Dict[str, str] = {}
    entry_pattern = re.compile(r'^\s*"([a-z0-9_]+)":\s*([A-Za-z0-9_]+)\(\),\s*$')
    for line in lines[start_idx + 1 : end_idx]:
        match = entry_pattern.match(line)
        if match:
            entry_map[match.group(1)] = match.group(2)
    entry_map[model_id] = class_name

    entry_lines = [f'    "{name}": {entry_map[name]}(),' for name in sorted(entry_map.keys())]
    new_lines = lines[: start_idx + 1] + entry_lines + lines[end_idx:]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return path


def build_pipeline_config(
    model_id: str = "m8_xgb",
    include_training: bool = True,
) -> dict[str, Any]:
    model = _validate_model_id(model_id)

    inference_cfg = deepcopy(DEFAULT_CONFIG)
    inference_cfg["model"]["selected_model"] = model
    inference_cfg["model"]["primary_model"] = model
    inference_cfg["model"]["enabled_models"] = [model]

    payload: dict[str, Any] = {
        "pipeline_schema_version": "1.0.0",
        "pynrpf_inference": inference_cfg,
    }

    if include_training:
        if model != "m8_xgb":
            raise ValueError(
                "Training config generation currently supports only model_id='m8_xgb'. "
                "Use include_training=False for other models."
            )
        training_cfg = deepcopy(DEFAULT_TRAINING_CONFIG)
        training_cfg["model_id"] = model
        payload["pynrpf_training"] = training_cfg

    return payload


def generate_pipeline_config(
    output_path: str | Path,
    model_id: str = "m8_xgb",
    include_training: bool = True,
    overwrite: bool = False,
) -> str:
    config_payload = build_pipeline_config(model_id=model_id, include_training=include_training)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Config file already exists: {path}. Set overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    return str(path)


def generate_model_scaffold(
    model_id: str,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    include_tests: bool = True,
    include_pipeline_config: bool = True,
) -> Dict[str, str]:
    model = _validate_model_id(model_id)
    root = Path(output_dir)
    created: Dict[str, str] = {}

    plugin_path = root / "src" / "pynrpf" / "plugins" / f"{model}.py"
    _write_text(plugin_path, _plugin_template(model), overwrite=overwrite)
    created["plugin_file"] = str(plugin_path)

    created["plugins_init_file"] = str(_ensure_plugins_init_wiring(root, model))
    created["registry_file"] = str(_ensure_registry_wiring(root, model))

    if include_tests:
        test_path = root / "tests" / f"test_{model}_plugin.py"
        _write_text(test_path, _test_template(model), overwrite=overwrite)
        created["test_file"] = str(test_path)

    if include_pipeline_config:
        cfg_path = root / "config" / f"pynrpf_pipeline_{model}.yaml"
        created["config_file"] = generate_pipeline_config(
            output_path=cfg_path,
            model_id=model,
            include_training=(model == "m8_xgb"),
            overwrite=overwrite,
        )

    return created
