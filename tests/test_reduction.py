import numpy as np
import pandas as pd
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.reduction import (
    apply_scaling,
    compute_scale_factors,
    finalize,
    mark_stitch_points,
    normalize_scan,
    reduce,
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
    # sam_z moves at index 2 -> cutoff 3 -> i0 = mean of first 4 (inclusive)
    i0 = np.mean([100.0, 100.0, 50.0, 25.0])
    np.testing.assert_allclose(out["R"].to_numpy(), df["counts_refl"] / i0)
    assert out["i0_mask"].tolist() == [1, 1, 1, 0]


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
    # sam_z moves at index 2 -> i0 region is indices 0,1,2 (theta 0,0,1);
    # reflectivity points are theta 2 and 3.
    assert len(out) == 2
    assert out["sam_th"].tolist() == [2, 3]
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
