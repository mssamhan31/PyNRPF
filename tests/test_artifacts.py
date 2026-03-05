from pathlib import Path

from pynrpf.artifacts import (
    _to_local_path,
    load_artifact_bundle,
    save_artifact_bundle,
    save_versioned_artifact_bundle,
)


def test_save_and_load_artifact_bundle(tmp_path: Path) -> None:
    bundle = {"bundle_schema": "test.v1", "value": 42}
    target = tmp_path / "bundle.pkl"
    saved = save_artifact_bundle(bundle, target)

    loaded = load_artifact_bundle(saved)
    assert loaded["bundle_schema"] == "test.v1"
    assert loaded["value"] == 42


def test_save_versioned_artifact_bundle(tmp_path: Path) -> None:
    bundle = {
        "bundle_schema": "pynrpf.m8_xgb.bundle.v2",
        "model_name": "m8_xgb",
        "created_at_utc": "2026-03-05T00:00:00+00:00",
        "xgb1_day": {"feature_columns": ["f1"], "threshold": 0.5},
        "xgb2_timestamp": {"feature_columns": ["f2"], "threshold": 0.7},
    }
    saved = save_versioned_artifact_bundle(
        bundle=bundle,
        base_location=tmp_path,
        model_name="m8_xgb",
        timestamp_utc="20260305T000000Z",
    )

    artifact_path = Path(saved["resolved_artifact_path"])
    manifest_path = Path(saved["resolved_manifest_path"])
    assert artifact_path.exists()
    assert manifest_path.exists()
    assert "m8_xgb" in saved["artifact_dir_uri"]
    assert "20260305T000000Z" in saved["artifact_dir_uri"]

    loaded = load_artifact_bundle(saved["artifact_uri"])
    assert loaded["bundle_schema"] == "pynrpf.m8_xgb.bundle.v2"


def test_uri_resolution_supports_dbfs_and_volumes_aliases() -> None:
    dbfs_path = str(_to_local_path("dbfs:/mnt/models/m8/bundle.pkl")).replace("\\", "/")
    dbfs_volume_alias = str(_to_local_path("dbfs:/Volumes/cat/sch/vol/bundle.pkl")).replace(
        "\\", "/"
    )
    volume_path = str(_to_local_path("/Volumes/cat/sch/vol/bundle.pkl")).replace("\\", "/")

    assert dbfs_path.endswith("/dbfs/mnt/models/m8/bundle.pkl")
    assert dbfs_volume_alias.endswith("/Volumes/cat/sch/vol/bundle.pkl")
    assert volume_path.endswith("/Volumes/cat/sch/vol/bundle.pkl")
