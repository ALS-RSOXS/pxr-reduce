"""Tests for diagnostic plots."""

from __future__ import annotations

import pytest

from pxr_reduce import diagnostics, reduction
from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader


def test_annotate_adds_working_columns(processed_loader_factory):
    loader = processed_loader_factory()
    ann = reduction.annotate(loader.data, loader.config)
    for col in ["R", "scale", "scale_err", "mark", "num_stitch_points",
                "failed_stitch_mask", "counts_spot"]:
        assert col in ann.columns
    assert len(ann) == len(loader.data)


def test_counts_vs_theta_figure_renders(processed_loader_factory):
    loader = processed_loader_factory()
    ann = reduction.annotate(loader.data, loader.config)
    (energy, pol), group = next(iter(ann.groupby(["energy", "polarization"])))
    fig = diagnostics.counts_vs_theta_figure(
        group, sample="S", energy=float(energy), pol=float(pol)
    )
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert "counts_spot" in ax.get_ylabel()


def test_beam_track_figure_bounds_and_inverted_y(processed_loader_factory):
    loader = processed_loader_factory()
    fig = diagnostics.beam_track_figure(loader, sample="S")
    ax = fig.axes[0]
    # Axes bounded to the trimmed image; y inverted (image convention).
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    xlo, _ = ax.get_xlim()
    assert xlo == 0


def test_save_diagnostics_writes_expected_files(processed_loader_factory, tmp_path):
    loader = processed_loader_factory()
    out = tmp_path / "diag"
    paths = diagnostics.save_diagnostics(loader, out)
    assert any(p.name == "beam_track.png" for p in paths)
    assert any(p.name.startswith("counts_vs_theta_") for p in paths)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_save_diagnostics_dry_run_writes_nothing(processed_loader_factory, tmp_path):
    loader = processed_loader_factory()
    out = tmp_path / "diag"
    paths = diagnostics.save_diagnostics(loader, out, dry_run=True)
    assert paths  # names are still returned
    assert not out.exists()


def test_save_diagnostics_requires_processed(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    files = sorted(folder.glob("*.fits"))
    loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
    with pytest.raises(RuntimeError):
        diagnostics.save_diagnostics(loader, tmp_path / "d")
