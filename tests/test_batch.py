"""Tests for batch reduction orchestration."""

from __future__ import annotations

import pytest

from pxr_reduce.batch import plan_batch, reduce_sample, run_batch, sample_files
from pxr_reduce.config import ReductionConfig
from pxr_reduce.run_config import RunConfig


def _make_scan(fits_writer, beam_image, frame_header, folder, sample, scan_id):
    """Write a 6-frame scan (2 i0 + 4 reflectivity) with real-style filenames."""
    folder.mkdir(parents=True, exist_ok=True)
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    for frame, (pk, z, th) in enumerate(zip(peaks, sam_z, sam_th)):
        path = folder / f"{sample}_{scan_id}-{frame:05d}.fits"
        fits_writer(path, beam_image(pk), frame_header(th, z))


@pytest.fixture
def beamtime(tmp_path, fits_writer, frame_builders):
    beam_image, frame_header = frame_builders
    parent = tmp_path / "beamtime"
    _make_scan(fits_writer, beam_image, frame_header, parent / "s1", "GlassA", 90001)
    _make_scan(fits_writer, beam_image, frame_header, parent / "s2", "GlassA", 90002)
    return parent


def _config(parent, results, samples, **kw):
    reduction = ReductionConfig(
        roi_height=9, roi_width=9, trim_x=2, trim_y=2, dark_pix_offset=5,
        new_scan_marker=3.0,  # split the two pooled scans (4 deg drop between them)
    )
    return RunConfig(
        parent_dir=parent, results_root=results, reduction=reduction, samples=samples,
        **kw,
    )


def test_sample_files_pools_scans_in_order(beamtime):
    cfg = _config(beamtime, beamtime / "out", {"GlassA": [90001, 90002]})
    files = sample_files(cfg, [90001, 90002])
    assert len(files) == 12
    # scan 90001's frames come before 90002's.
    assert all("90001" in p.name for p in files[:6])
    assert all("90002" in p.name for p in files[6:])


def test_plan_batch_counts_without_processing(beamtime):
    cfg = _config(beamtime, beamtime / "out", {"GlassA": [90001, 90002]})
    plan = plan_batch(cfg)
    assert plan[0]["sample"] == "GlassA"
    assert plan[0]["n_files"] == 12
    assert plan[0]["output"].name == "GlassA.dat"


def test_reduce_sample_single_scan_writes_dat(beamtime, tmp_path):
    out = tmp_path / "results"
    cfg = _config(beamtime, out, {"GlassA": [90001]})
    result = reduce_sample(cfg, "GlassA", progress=False)
    assert result["dat"].exists()
    assert result["dat"].name == "GlassA.dat"


def test_reduce_sample_multi_scan_pools_and_writes(beamtime, tmp_path):
    out = tmp_path / "results"
    cfg = _config(beamtime, out, {"Combo": [90001, 90002]})
    result = reduce_sample(cfg, "Combo", progress=False)
    assert result["dat"].exists()


def test_reduce_sample_no_files_raises(beamtime, tmp_path):
    cfg = _config(beamtime, tmp_path / "out", {"Missing": [99999]})
    with pytest.raises(FileNotFoundError):
        reduce_sample(cfg, "Missing", progress=False)


def test_run_batch_records_error_and_continues(beamtime, tmp_path):
    out = tmp_path / "results"
    cfg = _config(beamtime, out, {"GlassA": [90001], "Missing": [99999]})
    results = run_batch(cfg, progress=False)
    assert results["GlassA"]["dat"].exists()
    assert "error" in results["Missing"]


def test_run_batch_dry_run_writes_nothing(beamtime, tmp_path):
    out = tmp_path / "results"
    cfg = _config(beamtime, out, {"GlassA": [90001]})
    run_batch(cfg, progress=False, dry_run=True)
    assert not (out / "GlassA.dat").exists()
