from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .artifacts import save_versioned_artifact_bundle
from .config import ConfigInput, load_config
from .monitoring import build_operational_summary
from .registry import get_model
from .training_config import load_training_config
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


def train_m8_xgb(data: Any, config: ConfigInput) -> Dict[str, Any]:
    inference_cfg = load_config(config)
    training_cfg = load_training_config(config)

    _, pandas_df, _ = to_pandas_input(data)
    cleaned_df, _ = validate_dataframe(pandas_df, inference_cfg)

    cfg = deepcopy(inference_cfg)
    cfg["model"]["selected_model"] = "m8_xgb"
    cfg["training"] = deepcopy(training_cfg)

    m8_cfg = cfg.get("model", {}).setdefault("m8_xgb", {})
    xgb1_cfg = m8_cfg.setdefault("xgb1_day", {})
    xgb2_cfg = m8_cfg.setdefault("xgb2_timestamp", {})
    thresholds = training_cfg["thresholds"]
    xgb1_cfg["threshold"] = float(thresholds["xgb1_day"])
    xgb2_cfg["threshold"] = float(thresholds["xgb2_timestamp"])
    seed = int(training_cfg["random_seed"])
    xgb1_cfg["seed"] = seed
    xgb2_cfg["seed"] = seed

    plugin = get_model("m8_xgb")
    label_map = dict(training_cfg["labels"])
    bundle = plugin.train(cleaned_df, cfg, cfg["columns"], labels=label_map)

    training_meta = bundle.get("training_metadata", {})
    manifest = {
        "bundle_schema": bundle.get("bundle_schema"),
        "model_name": bundle.get("model_name"),
        "created_at_utc": bundle.get("created_at_utc"),
        "training_metadata": training_meta,
    }
    artifact_result = save_versioned_artifact_bundle(
        bundle=bundle,
        base_location=training_cfg["output"]["base_uri"],
        model_name="m8_xgb",
        manifest=manifest,
    )

    return {
        "bundle": bundle,
        "bundle_schema": bundle.get("bundle_schema"),
        "model": "m8_xgb",
        "artifact_dir_uri": artifact_result["artifact_dir_uri"],
        "artifact_uri": artifact_result["artifact_uri"],
        "manifest_uri": artifact_result["manifest_uri"],
        "validation_metrics": training_meta.get("validation_metrics", {}),
    }
