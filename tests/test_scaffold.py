from __future__ import annotations

from pathlib import Path

import yaml

from pynrpf.scaffold import (
    build_pipeline_config,
    generate_model_scaffold,
    generate_pipeline_config,
)


def test_generate_model_scaffold_creates_plugin_test_and_config(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "src" / "pynrpf" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / "__init__.py").write_text(
        'from .m7_dtr import M7DTRPlugin\n'
        'from .m8_xgb import M8XGBPlugin\n\n'
        '__all__ = ["M7DTRPlugin", "M8XGBPlugin"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "pynrpf" / "registry.py").write_text(
        "from __future__ import annotations\n\n"
        "from .plugins import M7DTRPlugin, M8XGBPlugin\n"
        "from .plugins.base import BaseModelPlugin\n\n"
        "_REGISTRY: dict[str, BaseModelPlugin] = {\n"
        '    "m7_dtr": M7DTRPlugin(),\n'
        '    "m8_xgb": M8XGBPlugin(),\n'
        "}\n",
        encoding="utf-8",
    )

    created = generate_model_scaffold("m9_custom", output_dir=tmp_path)

    plugin_path = Path(created["plugin_file"])
    test_path = Path(created["test_file"])
    init_path = Path(created["plugins_init_file"])
    registry_path = Path(created["registry_file"])
    config_path = Path(created["config_file"])

    assert plugin_path.exists()
    assert test_path.exists()
    assert init_path.exists()
    assert registry_path.exists()
    assert config_path.exists()
    assert "class M9CustomPlugin" in plugin_path.read_text(encoding="utf-8")
    assert "from .m9_custom import M9CustomPlugin" in init_path.read_text(encoding="utf-8")
    assert '"m9_custom": M9CustomPlugin(),' in registry_path.read_text(encoding="utf-8")


def test_generate_pipeline_config_writes_yaml(tmp_path: Path) -> None:
    out = tmp_path / "pipeline.yaml"
    path = generate_pipeline_config(out, model_id="m8_xgb", include_training=True)
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert "pynrpf_inference" in payload
    assert "pynrpf_training" in payload
    assert payload["pynrpf_inference"]["model"]["selected_model"] == "m8_xgb"


def test_build_pipeline_config_non_m8_training_rejected() -> None:
    try:
        build_pipeline_config("m9_custom", include_training=True)
    except ValueError as exc:
        assert "model_id='m8_xgb'" in str(exc)
        return
    raise AssertionError("Expected ValueError when include_training=True for non-m8 model.")
