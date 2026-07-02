"""Composable reduction stages: normalize -> mark stitches -> scale -> finalize.

Each stage is a pure function that takes the metadata/counts table plus a
:class:`~pxr_reduce.config.ReductionConfig` and returns an augmented copy. The
top-level :func:`reduce` runs them in order and lets callers skip the stitch/scale
stages for a fast "quick" reduction (see ``apply_scale``).

The stitch-detection and scale-factor algorithms are inherently sequential and
are ported faithfully from the original loader, with three changes:

* ``print`` diagnostics become ``logging`` calls.
* Per-scan groups are recombined with :func:`pandas.concat` instead of the
  original fragile ``igroup + i`` index arithmetic.
* The stitch scale-factor uncertainty is propagated into ``R_err`` (this was
  commented out in the original, so ``R_err`` previously ignored it).

Expected input columns: ``scan``, ``sam_th``, ``sam_z``, ``q``, ``energy``,
``polarization``, ``counts_refl``, ``counts_err``, ``counts_ratio``,
``is_saturated``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pxr_reduce import metadata
from pxr_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

# Motors whose motion marks a stitch boundary. ``sam_th`` is directional (a
# decrease marks a stitch); other motors use absolute motion.
STITCH_MOTORS: list[str] = ["sam_th"]

# sam_th must decrease by more than this (deg) to count as a stitch boundary.
_SAM_TH_STITCH_DROP = 0.2
# Frames to skip after marking a stitch, to avoid re-marking the same move.
_STITCH_SKIP_RESET = 2


def stitch_ratio_model(r: np.ndarray, scale: float) -> np.ndarray:
    """Model ``y = scale * r`` for fitting the stitch scale factor.

    Signature follows the :func:`scipy.optimize.curve_fit` convention
    (independent variable first, then parameters).

    Args:
        r: Independent (pre-change) reflectivity values.
        scale: Scale factor (fit parameter).

    Returns:
        Scaled reflectivity.
    """
    return scale * r


def _fit_scale_factor(
    pre_r: np.ndarray, post_r: np.ndarray
) -> tuple[float, float]:
    """Fit the through-origin scale factor ``post = scale * pre``.

    Uses a one-parameter least-squares fit. The uncertainty is taken from the
    covariance estimate; when it cannot be estimated (e.g. a single overlap
    point leaves zero degrees of freedom) the error is reported as 0.0.

    Args:
        pre_r: Pre-change reflectivity values.
        post_r: Post-change reflectivity values at the same angles.

    Returns:
        A ``(scale, scale_err)`` tuple.
    """
    popt, pcov = curve_fit(stitch_ratio_model, pre_r, post_r, p0=[1.0])
    scale = float(popt[0])
    variance = float(pcov[0, 0])
    scale_err = float(np.sqrt(variance)) if np.isfinite(variance) else 0.0
    return scale, scale_err


def _apply_per_scan(
    df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Apply a per-scan function to each scan group and recombine.

    Args:
        df: Full table containing a ``scan`` column.
        fn: Function mapping one scan group (with a fresh index) to a result.

    Returns:
        The concatenated result with a clean RangeIndex.
    """
    groups = [fn(g.reset_index(drop=True)) for _, g in df.groupby("scan", sort=True)]
    return pd.concat(groups, ignore_index=True)


def normalize_scan(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Normalize one scan to its direct-beam intensity (I0).

    I0 is the mean reflected counts before the sample moves into the beam
    (detected via ``sam_z`` motion). Reflectivity ``R = counts_refl / I0`` with
    propagated uncertainty.

    Args:
        df: One scan group (fresh index).
        config: Reduction configuration (unused here but kept for signature
            consistency across stages).

    Returns:
        The group with ``R``, ``R_err``, and ``i0_mask`` columns added.
    """
    df = df.copy()
    try:
        move_positions = df.index[df["sam_z"].diff().abs() > metadata.SAM_Z_BEAM_MOVE]
        i0_cutoff = int(move_positions[0]) + 1
        i0 = df["counts_refl"].loc[:i0_cutoff].mean()
        i0_err = df["counts_err"].loc[:i0_cutoff].std()
    except IndexError:
        logger.warning(
            "No direct beam found for scan starting at fits_index %s; not normalizing.",
            df["fits_index"].iloc[0] if "fits_index" in df else "?",
        )
        i0, i0_err, i0_cutoff = 1.0, 0.0, 0

    if i0 == 0 or np.isnan(i0):
        logger.warning("I0 evaluated to %s; falling back to 1.0.", i0)
        i0, i0_err = 1.0, 0.0

    df["R"] = df["counts_refl"] / i0
    rel_counts = np.where(
        df["counts_refl"] != 0, df["counts_err"] / df["counts_refl"], 0.0
    )
    df["R_err"] = np.abs(df["R"]) * np.sqrt(rel_counts**2 + (i0_err / i0) ** 2)
    df["i0_mask"] = (df.index < i0_cutoff).astype(int)
    return df


def mark_stitch_points(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Mark the frames at which a stitch (overlap) boundary begins.

    Args:
        df: One scan group (fresh index).
        config: Reduction configuration (stitch-mark tolerance).

    Returns:
        The group with ``mark`` (1 at a boundary) and ``motor`` columns added.
    """
    df = df.copy()
    n = len(df)
    mark: list[int | None] = [None] * n
    motor_of: list[str | None] = [None] * n

    for motor in STITCH_MOTORS:
        values = df[motor].to_numpy()
        if motor == "sam_th":
            steps = np.diff(values) < -_SAM_TH_STITCH_DROP
        else:
            steps = np.diff(values)

        skip = False
        skip_count = 0
        for i, val in enumerate(steps):
            if skip:
                if skip_count <= _STITCH_SKIP_RESET:
                    skip_count += 1
                else:
                    skip = False
                    skip_count = 0
            elif abs(val) > config.stitch_mark_tol:
                if mark[i] is None:
                    mark[i + 1] = 1
                    motor_of[i + 1] = motor
                skip = True

    df["mark"] = mark
    df["motor"] = motor_of
    return df


def compute_scale_factors(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Compute cumulative stitch scale factors for one scan.

    At each marked boundary, matches overlapping ``sam_th`` points before and
    after the change (excluding saturated and low-SNR frames), fits the ratio of
    their reflectivities with a one-parameter least-squares fit, and accumulates
    the scale factor and its uncertainty from that point onward.

    Args:
        df: One scan group (fresh index) that has been marked and normalized.
        config: Reduction configuration (stitch cutoff, drop-failed policy).

    Returns:
        The group with ``scale``, ``scale_err``, ``num_stitch_points``, and
        ``failed_stitch_mask`` columns added.
    """
    df = df.copy()
    df["scale"] = 1.0
    df["scale_err"] = 0.0
    df["num_stitch_points"] = 0
    df["failed_stitch_mask"] = 0

    sam_th = df["sam_th"]
    prev_mark_index = 0

    for i in range(len(df)):
        if pd.isna(df["mark"].iloc[i]):
            continue

        sam_th_stitch = sam_th.iloc[i]
        # Number of consecutive repeated angles at the stitch point.
        repeat = 0
        for val in sam_th.iloc[i:]:
            if val == sam_th_stitch:
                repeat += 1
            else:
                break

        # Pre-change indices whose angle recurs after the change.
        ipre: list[int] = []
        post_values = sam_th.iloc[i + repeat :].values
        for j, val in enumerate(sam_th.iloc[prev_mark_index:i]):
            jj = j + prev_mark_index
            if val in post_values and jj not in ipre:
                if df["is_saturated"].iloc[jj]:
                    logger.info("Saturated pre-change stitch point dropped: %d", jj)
                else:
                    ipre.append(jj)

        # Post-change indices whose angle appears in ipre (until the next mark).
        ipost: list[int] = []
        pre_values = sam_th.iloc[ipre].values
        for j, val in enumerate(sam_th.iloc[i + repeat :]):
            jj = j + i + repeat
            if not pd.isna(df["mark"].iloc[jj]):
                break
            if val in pre_values and jj not in ipost:
                if df["is_saturated"].iloc[jj]:
                    logger.info("Saturated post-change stitch point dropped: %d", jj)
                else:
                    ipost.append(jj)

        stitch_pre = df[["sam_th", "R"]].iloc[ipre].loc[
            df["counts_ratio"] > config.stitch_cutoff
        ]
        stitch_post = df[["sam_th", "R"]].iloc[ipost].loc[
            df["counts_ratio"] > config.stitch_cutoff
        ]
        safe_values = list(set(stitch_pre["sam_th"]) & set(stitch_post["sam_th"]))
        df.loc[i, "num_stitch_points"] = len(safe_values)

        if len(safe_values) == 0:
            logger.warning(
                "Failed stitch at index %d (energy %s eV, theta %s); masking "
                "subsequent points in scan.",
                i,
                df["energy"].iloc[i],
                sam_th.iloc[i],
            )
            df.loc[i:, "failed_stitch_mask"] = 1
            break
        if len(safe_values) == 1:
            warnings.warn(
                f"Scan starting at fits_index "
                f"{df['fits_index'].iloc[0] if 'fits_index' in df else '?'} "
                f"only has one stitch point.",
                stacklevel=2,
            )

        # Pair pre/post by angle so the fit gets equal-length, aligned arrays.
        # Repeated measurements at the same angle are averaged.
        order = sorted(safe_values)
        pre_r = stitch_pre.groupby("sam_th")["R"].mean().loc[order].to_numpy()
        post_r = stitch_post.groupby("sam_th")["R"].mean().loc[order].to_numpy()
        scale_i, scale_err_i = _fit_scale_factor(pre_r, post_r)

        prev_scale = df["scale"].iloc[i:].to_numpy()
        prev_scale_err = df["scale_err"].iloc[i:].to_numpy()
        new_scale = prev_scale * scale_i
        rel_prev = np.where(prev_scale != 0, prev_scale_err / prev_scale, 0.0)
        rel_i = scale_err_i / scale_i if scale_i != 0 else 0.0
        new_scale_err = np.abs(new_scale) * np.sqrt(rel_prev**2 + rel_i**2)
        df.loc[i:, "scale"] = new_scale
        df.loc[i:, "scale_err"] = new_scale_err

        prev_mark_index = i

    return df


def apply_scaling(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Divide ``R`` by the cumulative scale factor and propagate its error.

    This is the step whose error propagation was commented out in the original
    loader; ``R_err`` now includes the stitch scale-factor uncertainty.

    Args:
        df: Table with ``R``, ``R_err``, ``scale``, ``scale_err`` columns.
        config: Reduction configuration (unused; kept for signature consistency).

    Returns:
        The table with scaled ``R`` and ``R_err``.
    """
    df = df.copy()
    scale = df["scale"].to_numpy()
    scale_err = df["scale_err"].to_numpy()
    r = df["R"].to_numpy()
    r_err = df["R_err"].to_numpy()

    r_scaled = np.where(scale != 0, r / scale, 0.0)
    rel_r = np.where(r != 0, r_err / r, 0.0)
    rel_scale = np.where(scale != 0, scale_err / scale, 0.0)
    df["R"] = r_scaled
    df["R_err"] = np.abs(r_scaled) * np.sqrt(rel_r**2 + rel_scale**2)
    return df


def finalize(
    df: pd.DataFrame, config: ReductionConfig, drop_duplicates: bool = True
) -> pd.DataFrame:
    """Select valid reduced points and optionally average duplicates.

    Drops direct-beam frames, saturated frames, non-positive reflectivity, and
    (when configured) failed-stitch points.

    Args:
        df: Fully reduced table.
        config: Reduction configuration (drop-failed policy).
        drop_duplicates: If True, average points sharing (sam_th, energy,
            polarization).

    Returns:
        Reduced dataset with columns ``scan, energy, polarization, sam_th, q, R,
        R_err``.
    """
    mask = df["i0_mask"] < 1
    mask &= ~df["is_saturated"].astype(bool)
    mask &= df["R"] > 0
    if config.drop_failed_stitch:
        mask &= df["failed_stitch_mask"] < 1

    columns = ["scan", "energy", "polarization", "sam_th", "q", "R", "R_err"]
    out = df.loc[mask, columns]
    if drop_duplicates:
        out = out.groupby(["sam_th", "energy", "polarization"], as_index=False).mean()
    return out.reset_index(drop=True)


def reduce(
    df: pd.DataFrame,
    config: ReductionConfig,
    *,
    apply_scale: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Run the full reduction pipeline.

    Args:
        df: Metadata/counts table (see module docstring for required columns).
        config: Reduction configuration.
        apply_scale: If False, skip stitch detection and scaling (quick mode) —
            reflectivity is I0-normalized only. Useful for fast previews that
            avoid stitch-overlap pitfalls.
        drop_duplicates: Average points sharing (sam_th, energy, polarization).

    Returns:
        The reduced 1D dataset.
    """
    if df["is_saturated"].astype(bool).sum() > 0:
        warnings.warn(
            "The detector was likely saturated during collection; stitching may "
            "be impacted.",
            stacklevel=2,
        )

    df = _apply_per_scan(df, lambda g: normalize_scan(g, config))

    if apply_scale:
        df = _apply_per_scan(df, lambda g: mark_stitch_points(g, config))
        df = _apply_per_scan(df, lambda g: compute_scale_factors(g, config))
        df = apply_scaling(df, config)
    else:
        df = df.copy()
        df["scale"] = 1.0
        df["scale_err"] = 0.0
        df["failed_stitch_mask"] = 0

    return finalize(df, config, drop_duplicates=drop_duplicates)
