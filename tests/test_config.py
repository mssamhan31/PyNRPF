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
