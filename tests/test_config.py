from pynrpf.config import load_config


def test_load_config_adapts_legacy_run_yaml_shape() -> None:
    legacy = {
        "data": {
            "interval_minutes": 15,
            "columns": {
                "site": "site_id",
                "ts": "ts",
                "net_load": "mw",
                "solar": "solar_mw",
            },
        },
        "validation": {"enforce_interval_alignment": True},
        "m7_threshold": {"min_threshold": 0.1, "min_threshold_both": 0.2},
        "m8_xgb": {"xgb1_day": {"threshold": 0.6}, "xgb2_timestamp": {"threshold": 0.7}},
    }

    cfg = load_config(legacy)
    assert cfg["columns"]["site"] == "site_id"
    assert cfg["columns"]["timestamp"] == "ts"
    assert cfg["columns"]["net_load"] == "mw"
    assert cfg["columns"]["solar"] == "solar_mw"
    assert cfg["runtime"]["interval_minutes"] == 15
    assert cfg["model"]["m7_threshold"]["min_threshold"] == 0.1
    assert cfg["model"]["m8_xgb"]["xgb2_timestamp"]["threshold"] == 0.7


def test_load_config_extracts_pynrpf_inference_block() -> None:
    full_cfg = {
        "pipeline_schema_version": "1.0.0",
        "tables": {"inputs": {"ignored": "value"}},
        "pynrpf_inference": {
            "columns": {
                "site": "display_excel_name",
                "timestamp": "TS",
                "net_load": "MW",
                "solar": "Solar_MW",
            },
            "runtime": {"interval_minutes": 15, "strict_validation": True},
            "model": {"selected_model": "m7_dtr"},
        },
    }
    cfg = load_config(full_cfg)

    assert cfg["columns"]["site"] == "display_excel_name"
    assert cfg["columns"]["timestamp"] == "TS"
    assert cfg["model"]["selected_model"] == "m7_dtr"
