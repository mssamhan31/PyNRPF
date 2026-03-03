from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .artifacts import save_artifact_bundle
from .config import ConfigInput, load_config
from .monitoring import build_operational_summary
from .registry import get_model
from .validation import from_pandas_output, to_pandas_input, validate_dataframe


def run_inference(data: Any, config: ConfigInput) -> Dict[str, Any]:
    cfg = load_config(config)
    input_kind, pandas_df, spark_session = to_pandas_input(data)
    cleaned_df, dq_summary = validate_dataframe(pandas_df, cfg)

    model_name = cfg["model"]["selected_model"]
    plugin = get_model(model_name)
    result_df = plugin.run_inference(cleaned_df, cfg, cfg["columns"])

    summary = build_operational_summary(result_df, cfg, model_name, dq_summary)
    output = from_pandas_output(result_df, input_kind, spark_session)
    return {
        "data": output,
        "summary": summary,
        "model": model_name,
        "input_type": input_kind,
    }


def train_model(
    data: Any,
    config: ConfigInput,
    labels: Optional[Mapping[str, str]] = None,
    save_to: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config)
    _, pandas_df, _ = to_pandas_input(data)
    cleaned_df, _ = validate_dataframe(pandas_df, cfg)

    model_name = cfg["model"]["selected_model"]
    plugin = get_model(model_name)

    label_map = dict(labels) if labels is not None else {}
    if not label_map:
        from_cfg = cfg.get("model", {}).get("training_labels", {})
        if isinstance(from_cfg, dict):
            label_map = dict(from_cfg)

    bundle = plugin.train(cleaned_df, cfg, cfg["columns"], labels=label_map)

    saved_path = None
    if save_to:
        saved_path = save_artifact_bundle(bundle, save_to)

    return {
        "bundle": bundle,
        "saved_path": saved_path,
        "model": model_name,
    }
