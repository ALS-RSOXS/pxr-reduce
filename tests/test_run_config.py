"""Tests for the TOML run configuration."""

from __future__ import annotations

import pytest

from pxr_reduce.run_config import (
    RunConfig,
    load_run_config,
    resolve_config_path,
    run_config_to_toml_str,
    validate_run_config,
    write_default_config,
)

_TOML = """
[paths]
parent_dir = 'C:/data/beamtime'
results_root = 'out'
scan_number_width = 5

[tracking]
tracker = 'simple'
search_radius = 45

[export]
angle_decimals = 4
plots = false

[reduction]
roi_height = 30
trim_x = 12
stitch_condition_columns = ['hos', 'exposure']

[samples]
B1A1_NEdge_XRR = [89854, 89855]
B1A1_XRR_P100 = [17344]
"""


def test_load_applies_over_defaults(tmp_path):
    path = tmp_path / "reduction_config.toml"
    path.write_text(_TOML)
    cfg = load_run_config(path)
    assert cfg.parent_dir.name == "beamtime"
    assert cfg.tracker == "simple"
    assert cfg.search_radius == 45
    assert cfg.plots is False
    assert cfg.reduction.roi_height == 30
    assert cfg.reduction.trim_x == 12
    assert cfg.reduction.stitch_condition_columns == ("hos", "exposure")
    assert cfg.samples == {"B1A1_NEdge_XRR": [89854, 89855], "B1A1_XRR_P100": [17344]}


def test_load_none_returns_defaults():
    cfg = load_run_config(None)
    assert cfg == RunConfig()


def test_toml_round_trip(tmp_path):
    path = tmp_path / "reduction_config.toml"
    path.write_text(_TOML)
    cfg = load_run_config(path)
    out = tmp_path / "roundtrip.toml"
    out.write_text(run_config_to_toml_str(cfg))
    reloaded = load_run_config(out)
    assert reloaded.samples == cfg.samples
    assert reloaded.reduction == cfg.reduction
    assert reloaded.parent_dir == cfg.parent_dir
    assert reloaded.tracker == cfg.tracker


def test_invalid_tracker_rejected():
    with pytest.raises(ValueError):
        RunConfig(tracker="fancy")


def test_write_default_config_and_reload(tmp_path):
    dest = write_default_config(tmp_path / "starter")
    assert dest.suffix == ".toml"
    assert dest.exists()
    # The bundled template must itself be valid and loadable.
    load_run_config(dest)
    with pytest.raises(FileExistsError):
        write_default_config(dest)


def test_resolve_config_path_prefers_explicit_then_cwd(tmp_path, monkeypatch):
    explicit = tmp_path / "custom.toml"
    explicit.write_text(_TOML)
    assert resolve_config_path(explicit) == explicit

    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None) is None  # no cwd config yet
    (tmp_path / "reduction_config.toml").write_text(_TOML)
    assert resolve_config_path(None) == tmp_path / "reduction_config.toml"


def test_validate_reports_missing_parent_and_samples(tmp_path):
    cfg = RunConfig(parent_dir=tmp_path / "nope", samples={})
    problems = validate_run_config(cfg)
    assert any("parent_dir" in p for p in problems)
    assert any("samples" in p for p in problems)
