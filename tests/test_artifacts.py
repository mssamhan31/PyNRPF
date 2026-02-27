from pathlib import Path

from pynrpf.artifacts import load_artifact_bundle, save_artifact_bundle


def test_save_and_load_artifact_bundle(tmp_path: Path) -> None:
    bundle = {"bundle_schema": "test.v1", "value": 42}
    target = tmp_path / "bundle.pkl"
    saved = save_artifact_bundle(bundle, target)

    loaded = load_artifact_bundle(saved)
    assert loaded["bundle_schema"] == "test.v1"
    assert loaded["value"] == 42
