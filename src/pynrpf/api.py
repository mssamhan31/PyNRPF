"""Public entry points for reverse power flow (RPF) detection and correction.

Inputs:  an interval-level pandas or Spark DataFrame of substation net load, plus
         an inference (and for training, a training) configuration block.
Outputs: for inference, the scored frame with RPF flags and corrected net load in
         megawatts, plus an operational summary; for training, a versioned
         artefact bundle on disk and its validation metrics.
Key steps: load and validate the configuration, coerce the input to pandas and
         validate its schema, dispatch to the selected model plugin, then rebuild
         the caller's frame type and summarise the run.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .artifacts import save_versioned_artifact_bundle
from .config import ConfigInput, load_config
from .monitoring import build_operational_summary
from .registry import get_model
from .training_config import load_training_config
from .validation import from_pandas_output, to_pandas_input, validate_dataframe


def _require_strict_validation_for_m8(cfg: dict[str, Any], operation: str) -> None:
    strict = bool(cfg.get("runtime", {}).get("strict_validation", True))
    if not strict:
        raise ValueError(
            f"m8_xgb {operation} requires runtime.strict_validation=true "
            "to prevent duplicate timestamp expansion."
        )


def run_inference(data: Any, config: ConfigInput) -> Dict[str, Any]:
    """Score interval net load readings for reverse power flow and correct them.

    Args:
        data: Interval-level pandas or Spark DataFrame. Must carry the four
            columns named under the config's ``columns`` block: site, timestamp,
            net load in megawatts, and solar generation in megawatts.
        config: Mapping, or path to a YAML file. A full pipeline config
            containing a ``pynrpf_inference`` block is also accepted.

    Returns:
        Dict with ``data`` (the scored frame, same type as the input, carrying
        the RPF interval and day flags, corrected net load in megawatts and a
        confidence score), ``summary`` (row counts and monitoring statistics),
        ``model`` (the resolved model id) and ``input_type`` (``"pandas"`` or
        ``"spark"``).

    Raises:
        ValueError: If ``m8_xgb`` is selected while strict validation is off.
        KeyError: If the configured model id is not registered.
    """
    cfg = load_config(config)
    model_name = cfg["model"]["selected_model"]
    if model_name == "m8_xgb":
        _require_strict_validation_for_m8(cfg, "inference")

    input_kind, pandas_df, spark_session = to_pandas_input(data)
    cleaned_df, dq_summary = validate_dataframe(pandas_df, cfg)

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
    """Train both stages of the ``m8_xgb`` model and write a versioned bundle.

    Trains ``xgb1_day`` (day-level classifier) and ``xgb2_timestamp``
    (interval-level classifier) from one labelled interval dataset, then saves
    the pair as a single pickle bundle with a JSON manifest alongside it.

    Args:
        data: Labelled interval-level pandas or Spark DataFrame, carrying the
            four inference columns plus the day and interval label columns named
            in the training config.
        config: Mapping, or path to a YAML file, containing both a
            ``pynrpf_inference`` and a ``pynrpf_training`` block.

    Returns:
        Dict with ``bundle`` (the in-memory artefact), ``bundle_schema``,
        ``artifact_uri`` (the bundle path to feed back into inference),
        ``artifact_dir_uri``, ``manifest_uri`` and ``validation_metrics`` for
        both stages.

    Raises:
        ValueError: If strict validation is off, or the training split window or
            threshold values are invalid.
        KeyError: If a required label column is missing from the config.
    """
    inference_cfg = load_config(config)
    training_cfg = load_training_config(config)
    _require_strict_validation_for_m8(inference_cfg, "training")

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
