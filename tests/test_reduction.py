import numpy as np
import pandas as pd
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.reduction import (
    StitchFitError,
    _fit_scale_factor,
    annotate,
    apply_scaling,
    compute_scale_factors,
    diagnose_stitches,
    finalize,
    mark_stitch_points,
    normalize_scan,
    reduce,
    summarize_stitches,
)


def _scan_table(
    sam_th,
    sam_z,
    counts_refl,
    *,
    scan=0,
    counts_err=None,
    counts_ratio=2.0,
    is_saturated=False,
    energy=250.0,
    polarization=100.0,
):
    n = len(sam_th)
    if counts_err is None:
        counts_err = [1.0] * n
    return pd.DataFrame(
        {
            "fits_index": list(range(n)),
            "scan": [scan] * n,
            "sam_th": sam_th,
            "sam_z": sam_z,
            "q": np.linspace(0.01, 0.05, n),
            "energy": [energy] * n,
            "polarization": [polarization] * n,
            "counts_refl": counts_refl,
            "counts_err": counts_err,
            "counts_ratio": [counts_ratio] * n if np.isscalar(counts_ratio) else counts_ratio,
            "is_saturated": [is_saturated] * n,
        }
    )


def test_normalize_scan_sets_r_and_i0_mask():
    df = _scan_table(
        sam_th=[0, 0, 1, 2],
        sam_z=[0, 0, 1, 1],
        counts_refl=[100.0, 100.0, 50.0, 25.0],
    )
    out = normalize_scan(df, ReductionConfig())
    # sam_z moves at index 2 -> direct-beam frames are indices 0,1 (before the
    # move); i0 = mean(100, 100) = 100, so low-angle R normalizes to 1.
    np.testing.assert_allclose(out["R"].to_numpy(), [1.0, 1.0, 0.5, 0.25])
    assert out["i0_mask"].tolist() == [1, 1, 0, 0]


def test_normalize_scan_excludes_saturated_direct_beam():
    # First two frames are direct beam; the first is saturated (clipped low) and
    # must be excluded from i0 so R is not under-normalized.
    df = _scan_table(
        sam_th=[0, 0, 1, 2],
        sam_z=[0, 0, 1, 1],
        counts_refl=[20.0, 100.0, 50.0, 25.0],  # frame 0 clipped low
    )
    df["is_saturated"] = [True, False, False, False]
    out = normalize_scan(df, ReductionConfig())
    # i0 uses only the unsaturated direct-beam frame (100), not the clipped 20.
    np.testing.assert_allclose(out["R"].to_numpy(), [0.2, 1.0, 0.5, 0.25])


def test_normalize_scan_no_direct_beam_falls_back():
    df = _scan_table(
        sam_th=[1, 2, 3],
        sam_z=[1, 1, 1],  # no movement
        counts_refl=[10.0, 5.0, 2.0],
    )
    out = normalize_scan(df, ReductionConfig())
    # i0 falls back to 1.0 -> R == counts_refl
    np.testing.assert_allclose(out["R"].to_numpy(), [10.0, 5.0, 2.0])
    assert out["i0_mask"].tolist() == [0, 0, 0]


def test_apply_scaling_propagates_scale_error():
    df = pd.DataFrame(
        {
            "R": [10.0, 10.0],
            "R_err": [1.0, 1.0],  # 10% rel
            "scale": [2.0, 2.0],
            "scale_err": [0.2, 0.2],  # 10% rel
        }
    )
    out = apply_scaling(df, ReductionConfig())
    np.testing.assert_allclose(out["R"].to_numpy(), [5.0, 5.0])
    # combined relative error sqrt(0.1^2 + 0.1^2) => R_err = 5 * sqrt(0.02)
    np.testing.assert_allclose(out["R_err"].to_numpy(), 5.0 * np.sqrt(0.02))


def test_apply_scaling_without_scale_error_matches_plain_ratio():
    df = pd.DataFrame(
        {"R": [10.0], "R_err": [1.0], "scale": [2.0], "scale_err": [0.0]}
    )
    out = apply_scaling(df, ReductionConfig())
    # with zero scale error, R_err just scales by 1/scale
    np.testing.assert_allclose(out["R_err"].to_numpy(), [0.5])


def test_reduce_single_segment_end_to_end():
    df = _scan_table(
        sam_th=[0, 0, 1, 2, 3],
        sam_z=[0, 0, 1, 1, 1],
        counts_refl=[100.0, 100.0, 50.0, 25.0, 12.0],
    )
    out = reduce(df, ReductionConfig())
    assert {"sam_th", "q", "R", "R_err"}.issubset(out.columns)
    # sam_z moves at index 2 -> direct beam is indices 0,1 (theta 0,0);
    # reflectivity points are theta 1, 2, 3.
    assert len(out) == 3
    assert out["sam_th"].tolist() == [1, 2, 3]
    assert (out["R"] > 0).all()


def test_reduce_quick_mode_skips_scaling():
    df = _scan_table(
        sam_th=[0, 0, 1, 2, 3],
        sam_z=[0, 0, 1, 1, 1],
        counts_refl=[100.0, 100.0, 50.0, 25.0, 12.0],
    )
    full = reduce(df, ReductionConfig(), apply_scale=True)
    quick = reduce(df, ReductionConfig(), apply_scale=False)
    # single-segment scan has no stitches, so quick == full
    np.testing.assert_allclose(
        quick.sort_values("sam_th")["R"].to_numpy(),
        full.sort_values("sam_th")["R"].to_numpy(),
    )


def test_compute_scale_factors_fits_overlap():
    # Segment 1 angles 1,2,3 then drop back to 1,2,3,4 (overlap at 2 and 3).
    # Post-change R is exactly 2x pre-change -> expected scale factor 2.0.
    sam_th = [1, 2, 3, 1, 2, 3, 4]
    r = [9.9, 0.5, 0.25, 9.9, 1.0, 0.5, 0.2]  # th=2: 0.5->1.0, th=3: 0.25->0.5
    df = pd.DataFrame(
        {
            "fits_index": list(range(7)),
            "scan": [0] * 7,
            "sam_th": sam_th,
            "energy": [250.0] * 7,
            "R": r,
            "R_err": [0.01] * 7,
            "counts_ratio": [2.0] * 7,
            "is_saturated": [False] * 7,
        }
    )
    cfg = ReductionConfig()
    marked = mark_stitch_points(df, cfg)
    assert marked["mark"].iloc[3] == 1  # boundary at the theta drop
    scaled = compute_scale_factors(marked, cfg)
    # scale applied from the boundary onward, ~2.0
    assert scaled["scale"].iloc[6] == pytest.approx(2.0, rel=1e-6)
    assert scaled["scale"].iloc[0] == 1.0
    # two overlap points -> covariance estimable -> finite non-negative error
    assert np.isfinite(scaled["scale_err"].iloc[6])
    assert scaled["scale_err"].iloc[6] >= 0.0
    assert scaled["failed_stitch_mask"].iloc[6] == 0


def test_compute_scale_factors_with_repeated_overlap_angle():
    # Angle 2 is measured twice in the pre segment but once in post. The old code
    # produced mismatched-length arrays for the fit; pairing by angle fixes it.
    sam_th = [1, 2, 2, 3, 1, 2, 3, 4]
    r = [9.9, 0.5, 0.5, 0.25, 9.9, 1.0, 0.5, 0.2]  # post = 2x pre at angles 2, 3
    df = pd.DataFrame(
        {
            "fits_index": list(range(8)),
            "scan": [0] * 8,
            "sam_th": sam_th,
            "energy": [250.0] * 8,
            "R": r,
            "R_err": [0.01] * 8,
            "counts_ratio": [2.0] * 8,
            "is_saturated": [False] * 8,
        }
    )
    cfg = ReductionConfig()
    marked = mark_stitch_points(df, cfg)
    scaled = compute_scale_factors(marked, cfg)  # must not raise
    assert scaled["scale"].iloc[7] == pytest.approx(2.0, rel=1e-6)
    assert scaled["failed_stitch_mask"].iloc[7] == 0


def _stitch_scan_with_conditions():
    """Two-segment scan with i0 frames, a sub-0.2 back-step, and an exposure change.

    i0 (idx 0,1) -> segment 1 at exposure 1.0 (idx 2,3,4) -> stitch back into
    overlap with exposure 5.0 (idx 5,6,7,8). Post R is 2x pre at the overlap
    angles, so the expected scale is 2.0. The back-step (0.15 -> 0.05) is smaller
    than the old 0.2 deg threshold, which would have missed it.
    """
    sam_th = [0.0, 0.0, 0.05, 0.10, 0.15, 0.05, 0.10, 0.15, 0.20]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    exposure = [0.5, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
    # counts_refl / i0(=100) gives R; post = 2x pre at angles 0.10 and 0.15.
    counts_refl = [100.0, 100.0, 99.0, 50.0, 25.0, 99.0, 100.0, 50.0, 20.0]
    return pd.DataFrame(
        {
            "fits_index": list(range(9)),
            "scan": [0] * 9,
            "sam_th": sam_th,
            "sam_z": sam_z,
            "exposure": exposure,
            "q": np.linspace(0.01, 0.05, 9),
            "energy": [250.0] * 9,
            "polarization": [100.0] * 9,
            "counts_refl": counts_refl,
            "counts_err": [1.0] * 9,
            "counts_ratio": [2.0] * 9,
            "is_saturated": [False] * 9,
        }
    )


def test_mark_detects_condition_change_below_theta_threshold():
    df = normalize_scan(_stitch_scan_with_conditions(), ReductionConfig())
    marked = mark_stitch_points(df, ReductionConfig())
    boundaries = [i for i in range(len(marked)) if marked["mark"].iloc[i] == 1]
    # Exactly one boundary, at the back-step/exposure change (index 5).
    assert boundaries == [5]
    assert "exposure" in marked["stitch_trigger"].iloc[5]
    assert "backstep" in marked["stitch_trigger"].iloc[5]


def test_mark_never_flags_i0_to_first_measurement():
    # The i0->first-measurement step can go either way from an angle offset and
    # must never be a boundary. Here it is a *forward* step but with a condition
    # (exposure) change from i0 to the reflection segment.
    df = normalize_scan(_stitch_scan_with_conditions(), ReductionConfig())
    marked = mark_stitch_points(df, ReductionConfig())
    # i0 frames (0,1) and the first reflection frame (2) are never marked.
    assert pd.isna(marked["mark"].iloc[0])
    assert pd.isna(marked["mark"].iloc[1])
    assert pd.isna(marked["mark"].iloc[2])


def test_mark_collapses_settling_repeats_to_one_boundary():
    # A boundary followed by 3 settling repeats at the backed-up angle, then the
    # overlap is re-measured forward. Only one boundary should be marked.
    df = pd.DataFrame(
        {
            "fits_index": list(range(9)),
            "scan": [0] * 9,
            "sam_th": [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0],
            "i0_mask": [0] * 9,
            "exposure": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    marked = mark_stitch_points(df, ReductionConfig())
    assert [i for i in range(len(marked)) if marked["mark"].iloc[i] == 1] == [3]


def test_mark_falls_back_to_backstep_without_condition_columns():
    # No condition columns present -> pure sam_th back-step detection still works.
    sam_th = [1, 2, 3, 1, 2, 3, 4]
    df = pd.DataFrame(
        {
            "fits_index": list(range(7)),
            "scan": [0] * 7,
            "sam_th": sam_th,
            "i0_mask": [0] * 7,
        }
    )
    marked = mark_stitch_points(df, ReductionConfig())
    assert marked["mark"].iloc[3] == 1
    assert [i for i in range(7) if marked["mark"].iloc[i] == 1] == [3]


def test_diagnose_stitches_reports_boundary_details():
    diag = diagnose_stitches(_stitch_scan_with_conditions(), ReductionConfig())
    assert len(diag) == 1
    row = diag.iloc[0]
    assert row["num_stitch_points"] == 2  # angles 0.10 and 0.15 overlap
    assert row["scale"] == pytest.approx(2.0, rel=1e-6)
    assert not row["failed"]
    assert "exposure" in row["conditions_changed"]
    assert "5" in row["conditions_changed"]  # exposure 1 -> 5


def _i0_angle_collides_with_overlap():
    """Scan whose direct-beam frames sit at an angle re-measured after the stitch.

    i0 is taken at sam_th 0.20 (idx 0,1), which also appears in the post-boundary
    segment. Those i0 frames have R = 1.0 (sample out of the beam), so including
    them in the pre-change overlap drags the fitted scale to ~0.70; the reflectivity
    frames alone give exactly 2.0.
    """
    return pd.DataFrame(
        {
            "fits_index": list(range(9)),
            "scan": [0] * 9,
            "sam_th": [0.20, 0.20, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.40],
            "sam_z": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "exposure": [1.0] * 9,
            "q": np.linspace(0.01, 0.05, 9),
            "energy": [250.0] * 9,
            "polarization": [100.0] * 9,
            "counts_refl": [100.0, 100.0, 50.0, 25.0, 12.0, 100.0, 50.0, 24.0, 10.0],
            "counts_err": [1.0] * 9,
            "counts_ratio": [2.0] * 9,
            "is_saturated": [False] * 9,
        }
    )


def test_compute_scale_factors_excludes_i0_frames_from_overlap():
    cfg = ReductionConfig()
    df = normalize_scan(_i0_angle_collides_with_overlap(), cfg)
    marked = mark_stitch_points(df, cfg)
    assert [i for i in range(len(marked)) if marked["mark"].iloc[i] == 1] == [5]

    scaled = compute_scale_factors(marked, cfg)
    # Only the reflectivity frames at theta 0.20 and 0.30 are legitimate overlap.
    assert scaled["num_stitch_points"].iloc[5] == 2
    assert scaled["scale"].iloc[8] == pytest.approx(2.0, rel=1e-6)


def _failed_then_good_stitch_scan():
    """Scan with a first boundary that has no overlap and a second that does.

    i0 (idx 0,1) -> segment 0 (idx 2,3) -> boundary A at idx 4 from an exposure
    change with no re-measured angles (unstitchable) -> segment 1 (idx 4,5,6) ->
    boundary B at idx 7 stepping back into overlap at theta 0.40/0.50, where post R
    is exactly 2x pre R.
    """
    return pd.DataFrame(
        {
            "fits_index": list(range(11)),
            "scan": [0] * 11,
            "sam_th": [
                0.10, 0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60
            ],
            "sam_z": [0.0, 0.0] + [1.0] * 9,
            "exposure": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0],
            "q": np.linspace(0.01, 0.06, 11),
            "energy": [250.0] * 11,
            "polarization": [100.0] * 11,
            "counts_refl": [
                100.0, 100.0, 100.0, 50.0, 25.0, 12.5, 6.25, 50.0, 25.0, 12.5, 6.0
            ],
            "counts_err": [1.0] * 11,
            "counts_ratio": [2.0] * 11,
            "is_saturated": [False] * 11,
        }
    )


def test_compute_scale_factors_continues_after_failed_stitch():
    scaled = annotate(_failed_then_good_stitch_scan(), ReductionConfig())
    assert [i for i in range(len(scaled)) if scaled["mark"].iloc[i] == 1] == [4, 7]

    # Boundary A failed and says why; boundary B was still evaluated (no break).
    assert scaled["stitch_failed"].iloc[4] == 1
    assert "overlap" in scaled["stitch_fail_reason"].iloc[4]
    assert scaled["stitch_failed"].iloc[7] == 0
    assert scaled["num_stitch_points"].iloc[7] == 2
    assert scaled["scale"].iloc[10] == pytest.approx(2.0, rel=1e-6)


def test_failed_stitch_masks_only_from_the_failure_onward():
    scaled = annotate(_failed_then_good_stitch_scan(), ReductionConfig())
    # Points before the failure keep an established absolute scale; everything
    # from the failure on is tied to the unknown factor at boundary A.
    assert scaled["failed_stitch_mask"].iloc[:4].tolist() == [0, 0, 0, 0]
    assert (scaled["failed_stitch_mask"].iloc[4:] == 1).all()


def test_reduce_keeps_pre_failure_points_when_dropping_failed_stitches():
    out = reduce(_failed_then_good_stitch_scan(), ReductionConfig())
    # Segment 0 (theta 0.10, 0.20) survives; the unestablished tail is dropped.
    assert out["sam_th"].tolist() == [0.10, 0.20]


def test_diagnose_stitches_reports_fail_reason_and_scale_established():
    diag = diagnose_stitches(_failed_then_good_stitch_scan(), ReductionConfig())
    assert len(diag) == 2

    first, second = diag.iloc[0], diag.iloc[1]
    assert first["failed"]
    assert "overlap" in first["fail_reason"]
    assert not first["scale_established"]
    # The second boundary fitted cleanly but still inherits the unknown offset.
    assert not second["failed"]
    assert second["fail_reason"] == ""
    assert second["num_stitch_points"] == 2
    assert second["scale"] == pytest.approx(2.0, rel=1e-6)
    assert not second["scale_established"]


def _exposure_only_stitch(post_multiplier: float, *, spread: float = 0.0):
    """Two-segment scan whose only condition change is exposure.

    Reflectivity is already exposure-normalized, so a correct reduction fits a scale
    of 1.0 here. ``post_multiplier`` scales the post-boundary segment to simulate a
    wrong scale; ``spread`` perturbs the two overlap angles in opposite directions to
    simulate overlap points that disagree with each other.
    """
    pre = [50.0, 25.0]
    post = [50.0 * post_multiplier * (1 + spread), 25.0 * post_multiplier * (1 - spread)]
    return pd.DataFrame(
        {
            "fits_index": list(range(9)),
            "scan": [0] * 9,
            "sam_th": [0.05, 0.05, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.40],
            "sam_z": [0.0, 0.0] + [1.0] * 7,
            "exposure": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
            "q": np.linspace(0.01, 0.05, 9),
            "energy": [250.0] * 9,
            "polarization": [100.0] * 9,
            "counts_refl": [100.0, 100.0, 99.0] + pre + [99.0] + post + [10.0],
            "counts_err": [1.0] * 9,
            "counts_ratio": [2.0] * 9,
            "is_saturated": [False] * 9,
        }
    )


def test_exposure_only_stitch_expects_unit_scale_and_passes():
    scaled = annotate(_exposure_only_stitch(1.0), ReductionConfig())
    b = 5  # the boundary
    assert scaled["num_stitch_points"].iloc[b] == 2
    assert scaled["expected_scale"].iloc[b] == pytest.approx(1.0)
    assert scaled["overlap_rms_rel"].iloc[b] == pytest.approx(0.0, abs=1e-9)
    assert scaled["stitch_suspect"].iloc[b] == 0
    assert scaled["stitch_quality_note"].iloc[b] is None


def test_exposure_only_stitch_flags_scale_far_from_expected():
    # A 40% offset across an exposure-only boundary cannot be physical.
    scaled = annotate(_exposure_only_stitch(1.4), ReductionConfig())
    b = 5
    assert scaled["scale"].iloc[b] == pytest.approx(1.4, rel=1e-6)
    assert scaled["stitch_suspect"].iloc[b] == 1
    assert "expected" in scaled["stitch_quality_note"].iloc[b]
    # Suspect is diagnostic only: the scale is still applied and nothing is dropped.
    assert scaled["failed_stitch_mask"].iloc[b] == 0
    assert len(reduce(_exposure_only_stitch(1.4), ReductionConfig())) == 4


def test_overlap_points_that_disagree_are_flagged():
    # Same mean scale, but the two overlap angles disagree by +/-30%.
    scaled = annotate(_exposure_only_stitch(1.0, spread=0.30), ReductionConfig())
    b = 5
    assert scaled["overlap_rms_rel"].iloc[b] > 0.20
    assert scaled["stitch_suspect"].iloc[b] == 1
    assert "disagree" in scaled["stitch_quality_note"].iloc[b]


def test_stitch_quality_thresholds_are_configurable():
    df = _exposure_only_stitch(1.4)
    lenient = annotate(df, ReductionConfig(stitch_max_scale_deviation=0.50))
    assert lenient["stitch_suspect"].iloc[5] == 0
    strict = annotate(df, ReductionConfig(stitch_max_scale_deviation=0.01))
    assert strict["stitch_suspect"].iloc[5] == 1


def test_single_overlap_point_is_suspect_not_silently_precise():
    # The segment steps back to 0.20 but only theta 0.30 is actually re-measured
    # (the backed-up angle itself is excluded from the post window), so the scale is
    # exact by construction and its rms residual is meaningless -- it must not read
    # as a clean stitch.
    df = pd.DataFrame(
        {
            "fits_index": list(range(8)),
            "scan": [0] * 8,
            "sam_th": [0.05, 0.05, 0.10, 0.20, 0.30, 0.20, 0.30, 0.40],
            "sam_z": [0.0, 0.0] + [1.0] * 6,
            "exposure": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
            "q": np.linspace(0.01, 0.04, 8),
            "energy": [250.0] * 8,
            "polarization": [100.0] * 8,
            "counts_refl": [100.0, 100.0, 99.0, 50.0, 25.0, 50.0, 25.0, 10.0],
            "counts_err": [1.0] * 8,
            "counts_ratio": [2.0] * 8,
            "is_saturated": [False] * 8,
        }
    )
    scaled = annotate(df, ReductionConfig())
    b = 5
    assert scaled["num_stitch_points"].iloc[b] == 1
    assert np.isnan(scaled["overlap_rms_rel"].iloc[b])
    assert scaled["stitch_suspect"].iloc[b] == 1
    assert "one overlap point" in scaled["stitch_quality_note"].iloc[b]


def test_flux_changing_stitch_has_no_expected_scale():
    # A slits change alters the incident flux, so any scale is legitimate and the
    # expected-scale check must not fire.
    df = _exposure_only_stitch(1.4)
    df["slits_vert"] = [0.5] * 5 + [1.0] * 4
    scaled = annotate(df, ReductionConfig())
    b = 5
    assert "slits_vert" in scaled["stitch_trigger"].iloc[b]
    assert np.isnan(scaled["expected_scale"].iloc[b])
    assert scaled["stitch_suspect"].iloc[b] == 0


def test_summarize_stitches_partitions_outcomes():
    report = pd.DataFrame(
        {
            # A failed boundary must not also be counted as suspect.
            "failed": [False, False, True, True],
            "suspect": [False, True, True, False],
        }
    )
    assert summarize_stitches(report) == {
        "total": 4,
        "ok": 1,
        "suspect": 1,
        "failed": 2,
    }


@pytest.mark.parametrize("report", [None, pd.DataFrame()])
def test_summarize_stitches_handles_empty(report):
    assert summarize_stitches(report) == {
        "total": 0,
        "ok": 0,
        "suspect": 0,
        "failed": 0,
    }


def test_diagnose_stitches_reports_quality_columns():
    diag = diagnose_stitches(_exposure_only_stitch(1.4), ReductionConfig())
    row = diag.iloc[0]
    assert row["suspect"]
    assert "expected" in row["quality_note"]
    assert row["expected_scale"] == pytest.approx(1.0)
    assert row["overlap_rms_rel"] == pytest.approx(0.0, abs=1e-9)
    assert not row["failed"]
    assert row["scale_established"]


@pytest.mark.parametrize(
    "pre_r,post_r,reason",
    [
        ([], [], "no overlap"),
        ([1.0, np.nan], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, np.inf], "finite"),
        ([0.0, 0.0], [1.0, 2.0], "zero"),
        ([1.0, 2.0], [-1.0, -2.0], "non-positive"),
        ([1.0, 2.0], [0.0, 0.0], "non-positive"),
    ],
)
def test_fit_scale_factor_rejects_degenerate_input(pre_r, post_r, reason):
    with pytest.raises(StitchFitError, match=reason):
        _fit_scale_factor(np.asarray(pre_r, dtype=float), np.asarray(post_r, dtype=float))


def test_fit_scale_factor_returns_scale_for_clean_overlap():
    scale, scale_err = _fit_scale_factor(
        np.array([1.0, 0.5]), np.array([2.0, 1.0])
    )
    assert scale == pytest.approx(2.0)
    assert np.isfinite(scale_err) and scale_err >= 0.0


def test_apply_scaling_flags_degenerate_scale_instead_of_zeroing():
    df = pd.DataFrame(
        {
            "R": [10.0, 10.0, 10.0],
            "R_err": [1.0, 1.0, 1.0],
            "scale": [2.0, 0.0, -1.0],
            "scale_err": [0.0, 0.0, 0.0],
        }
    )
    out = apply_scaling(df, ReductionConfig())
    # A degenerate scale yields NaN (visible, and dropped by finalize) rather than
    # a plausible-looking 0.0 or a sign-flipped value.
    np.testing.assert_allclose(out["R"].to_numpy(), [5.0, np.nan, np.nan])
    np.testing.assert_allclose(out["R_err"].to_numpy(), [0.5, np.nan, np.nan])


def test_finalize_drops_saturated_and_nonpositive():
    df = _scan_table(
        sam_th=[1, 2, 3],
        sam_z=[1, 1, 1],
        counts_refl=[10.0, 5.0, 2.0],
    )
    df["R"] = [1.0, -0.5, 2.0]  # middle is non-positive
    df["R_err"] = [0.1, 0.1, 0.1]
    df["i0_mask"] = 0
    df["failed_stitch_mask"] = 0
    df.loc[2, "is_saturated"] = True  # last is saturated
    out = finalize(df, ReductionConfig(), drop_duplicates=False)
    assert len(out) == 1
    assert out["sam_th"].iloc[0] == 1
