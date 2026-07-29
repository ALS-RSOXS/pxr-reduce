"""Tests for the per-boundary stitch diagnostics folder."""

from __future__ import annotations

import numpy as np
import pytest

from pxr_reduce import stitch_diagnostics as sd
from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader

# theta -> beam peak at unit exposure; counts scale with exposure so an
# exposure-only stitch fits ~1.0 (as a correct reduction must).
_BASE_PEAK = {0.0: 10000.0, 1.0: 6000.0, 2.0: 3000.0, 3.0: 1500.0, 4.0: 700.0,
              5.0: 300.0}

# (sam_th, sam_z, exposure); i0 -> segment at exposure 1 -> back-step at exposure 5.
_FRAMES = [
    (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (3.0, 1.0, 1.0), (4.0, 1.0, 1.0),
    (2.0, 1.0, 5.0), (3.0, 1.0, 5.0), (4.0, 1.0, 5.0), (5.0, 1.0, 5.0),
]


@pytest.fixture
def stitch_loader(tmp_path, frame_builders, fits_writer):
    """Return a factory building a processed loader with a real stitch boundary.

    ``saturate`` names frame indices whose beam core is clipped to full scale. The
    core is clipped wholesale rather than as a small patch, because the dezinger
    median filter removes isolated hot pixels before saturation is ever tested.
    """
    beam_image, frame_header = frame_builders

    def _make(saturate: tuple[int, ...] = ()) -> PXRLoader:
        files = []
        for i, (th, z, exposure) in enumerate(_FRAMES):
            image = beam_image(_BASE_PEAK[th] * exposure)
            if i in saturate:
                image[25:36, 25:36] = 65535.0
            header = frame_header(th, z)
            header["EXPOSURE"] = exposure
            files.append(
                fits_writer(tmp_path / f"MF999A_{i}.fits", image, header)
            )
        loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
        loader.process()
        return loader

    return _make


def test_saturation_is_scoped_to_the_beam_roi(stitch_loader):
    loader = stitch_loader(saturate=(4,))
    data = loader.data.set_index("fits_index")
    assert bool(data.loc[4, "is_saturated"])
    assert data.loc[4, "n_sat_roi"] > 0
    # Every other frame is clean, and no frame reports dark-ROI saturation.
    others = data.drop(index=4)
    assert not others["is_saturated"].any()
    assert (others["n_sat_roi"] == 0).all()
    assert (data["n_sat_dark"] == 0).all()


def test_saturation_outside_the_roi_is_not_flagged(
    tmp_path, frame_builders, fits_writer
):
    # A saturated block far from the beam must not flag the frame: it never enters
    # counts_spot, and flagging it would drop a good point.
    beam_image, frame_header = frame_builders
    files = []
    for i, (th, z, exposure) in enumerate(_FRAMES):
        image = beam_image(_BASE_PEAK[th] * exposure)
        if i == 4:
            image[0:6, 0:6] = 65535.0  # corner, well outside the 9x9 beam ROI
        header = frame_header(th, z)
        header["EXPOSURE"] = exposure
        files.append(fits_writer(tmp_path / f"S_{i}.fits", image, header))
    loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
    loader.process()
    assert not loader.data["is_saturated"].any()
    assert (loader.data["n_sat_roi"] == 0).all()


def test_overlap_report_matches_the_points_actually_fitted(stitch_loader):
    loader = stitch_loader()
    report = loader.overlap_report()
    boundaries = loader.diagnose_stitches()
    assert len(boundaries) == 1

    # The report's used points must agree with the fit's own overlap count, since
    # both come from the same selection.
    used = report[report["used"]]
    assert used["sam_th"].nunique() == int(boundaries.iloc[0]["num_stitch_points"])
    assert (report.loc[report["used"], "reason"] == "").all()
    assert (report.loc[~report["used"], "reason"] != "").all()


def test_saturated_overlap_point_is_dropped_and_costs_its_partner(stitch_loader):
    # Frame 4 is the pre-side overlap point at theta 3. Saturating it must drop that
    # point and leave its post-side counterpart (frame 7) unpairable.
    report = stitch_loader(saturate=(4,)).overlap_report()
    by_index = report.set_index("fits_index")
    assert not by_index.loc[4, "used"]
    assert by_index.loc[4, "reason"].startswith("saturated")
    assert not by_index.loc[7, "used"]
    assert "partner dropped" in by_index.loc[7, "reason"]
    # Theta 4 still pairs, so the boundary survives with one overlap angle.
    assert by_index.loc[5, "used"] and by_index.loc[8, "used"]


def test_save_stitch_diagnostics_writes_scan_folder_and_summary(
    stitch_loader, tmp_path
):
    loader = stitch_loader(saturate=(4,))
    out = tmp_path / "stitch"
    paths = sd.save_stitch_diagnostics(loader, out)

    for path in paths:
        assert path.exists() and path.stat().st_size > 0
    assert (out / "stitch_summary.md").exists()
    assert (out / "dropped_points.md").exists()
    assert list((out / "scan_00").glob("boundary_*.png"))
    # Only saturated frames that cost a stitch point get an ROI image.
    assert (out / "scan_00" / "saturated" / "frame_00004_roi.png").exists()


def test_summary_names_the_file_and_reason_for_each_dropped_point(
    stitch_loader, tmp_path
):
    loader = stitch_loader(saturate=(4,))
    out = tmp_path / "stitch"
    sd.save_stitch_diagnostics(loader, out)
    text = (out / "stitch_summary.md").read_text(encoding="utf-8")

    assert "**Dropped overlap points**" in text
    assert "saturated (32 px in beam ROI)" in text
    assert "partner dropped" in text
    # The source FITS file and the ROI image are both named.
    assert "MF999A_4.fits" in text
    assert "scan_00/saturated/frame_00004_roi.png" in text
    # Saturation semantics are stated, so the report is self-explaining.
    assert "integrated beam ROI" in text
    # Markdown structure: headings, a table, and the figure embedded.
    assert text.startswith("# Stitch diagnostics")
    assert "### Boundary 01" in text
    assert "|---|" in text
    assert "![Boundary 01](scan_00/boundary_01.png)" in text


def test_dropped_points_markdown_carries_full_paths(stitch_loader, tmp_path):
    loader = stitch_loader(saturate=(4,))
    out = tmp_path / "stitch"
    sd.save_stitch_diagnostics(loader, out)
    text = (out / "dropped_points.md").read_text(encoding="utf-8")

    assert text.startswith("# Dropped stitch-overlap points")
    assert "| scan | boundary |" in text
    # Full source path, not just the basename.
    assert str(loader.path_for(4)) in text
    assert "saturated" in text and "partner dropped" in text
    # Only rejected candidates appear; the used ones (frames 5 and 8) do not.
    assert "| 5 |" not in text.replace(str(loader.path_for(4)), "")


def test_summary_lists_saturated_frames_that_are_not_stitch_points(
    stitch_loader, tmp_path
):
    # Frame 9 (theta 5, the last point) is never an overlap candidate. It must still
    # be reported, but without an ROI image.
    loader = stitch_loader(saturate=(9,))
    out = tmp_path / "stitch"
    sd.save_stitch_diagnostics(loader, out)
    text = (out / "stitch_summary.md").read_text(encoding="utf-8")

    assert "## Saturated frames" in text
    assert "MF999A_9.fits" in text
    assert "not a stitch overlap candidate" in text
    assert not (out / "scan_00" / "saturated" / "frame_00009_roi.png").exists()


def test_save_stitch_diagnostics_dry_run_writes_nothing(stitch_loader, tmp_path):
    loader = stitch_loader(saturate=(4,))
    out = tmp_path / "stitch"
    paths = sd.save_stitch_diagnostics(loader, out, dry_run=True)
    assert paths  # the intended targets are still reported
    assert not out.exists()


def test_save_stitch_diagnostics_requires_processed(
    tmp_path, frame_builders, fits_writer
):
    beam_image, frame_header = frame_builders
    # Two files minimum: the loader infers frame ordering from the filename sequence.
    files = [
        fits_writer(tmp_path / f"S_{i}.fits", beam_image(100.0), frame_header(1.0, 1.0))
        for i in range(2)
    ]
    loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
    with pytest.raises(RuntimeError):
        sd.save_stitch_diagnostics(loader, tmp_path / "stitch")


def test_boundary_figure_bottom_panel_plots_post_against_pre(stitch_loader):
    from pxr_reduce import reduction

    loader = stitch_loader(saturate=(4,))
    annotated = reduction.annotate(loader.data, loader.config)
    boundaries = reduction.diagnose_stitches(
        loader.data, loader.config, annotated=annotated
    )
    points = reduction.overlap_report(
        loader.data, loader.config, annotated=annotated
    )
    boundary = boundaries.iloc[0]
    group = annotated[annotated["scan"] == boundary["scan"]].reset_index(drop=True)

    fig = sd.boundary_figure(
        group,
        points[points["boundary_index"] == boundary["boundary_index"]],
        boundary,
        sample="S",
    )
    top, bottom = fig.axes[0], fig.axes[1]
    assert top.get_yscale() == "log"
    assert "sam_th" in top.get_xlabel()
    # Bottom panel is the fit itself: post-change R against pre-change R, with the
    # through-origin fit line, so both axes must include the origin.
    assert "pre-change R" in bottom.get_xlabel()
    assert "post-change R" in bottom.get_ylabel()
    assert bottom.get_xlim()[0] == 0.0
    assert bottom.get_ylim()[0] == 0.0
    labels = [t.get_text() for t in bottom.get_legend().get_texts()]
    assert any(label.startswith("fit: post =") for label in labels)
    assert any("overlap angles" in label for label in labels)


def test_saturated_roi_figure_marks_beam_and_saturated_pixels(stitch_loader):
    loader = stitch_loader(saturate=(4,))
    row = loader.data.set_index("fits_index", drop=False).loc[4]
    fig = sd.saturated_roi_figure(loader, 4, row)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("saturated" in label for label in labels)
    assert any("beam centre" in label for label in labels)
    assert "32 px in ROI" in ax.get_title()


@pytest.mark.parametrize("scan_id,expected", [(0, "scan_00"), (7, "scan_07")])
def test_scan_dir_name(scan_id, expected):
    assert sd._scan_dir_name(scan_id) == expected


def test_by_scan_tolerates_a_sample_with_no_boundaries():
    import pandas as pd

    # diagnose_stitches returns a column-less frame when nothing was detected.
    assert sd._by_scan(pd.DataFrame()) == []


def test_images_are_capped_and_the_shortfall_is_logged(stitch_loader, caplog):
    loader = stitch_loader(saturate=(4,))
    report = loader.overlap_report()
    annotated = loader.data
    with caplog.at_level("WARNING"):
        chosen = sd._images_to_write(annotated, report, max_images=0)
    assert chosen == []
    assert "only 0 ROI images" in caplog.text


def test_diagnostics_folder_includes_stitch_subfolder(stitch_loader, tmp_path):
    from pxr_reduce import diagnostics

    loader = stitch_loader(saturate=(4,))
    out = tmp_path / "diag"
    paths = diagnostics.save_diagnostics(loader, out)
    assert (out / "stitch" / "stitch_summary.md").exists()
    assert any("stitch" in str(p) for p in paths)


def test_expected_scale_holds_for_a_physical_exposure_stitch(stitch_loader):
    # Counts scale with exposure and R is exposure-normalized, so the fitted scale
    # must be ~1. Guards the expected-scale check against a false positive.
    boundaries = stitch_loader().diagnose_stitches()
    assert boundaries.iloc[0]["scale"] == pytest.approx(1.0, abs=0.02)
    assert np.isclose(boundaries.iloc[0]["expected_scale"], 1.0)
