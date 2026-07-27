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
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pxr_reduce import metadata
from pxr_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

# Tolerance for comparing pre-rounded metadata values (angles, conditions).
_FLOAT_EPS = 1e-9


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
    scan_id = df["fits_index"].iloc[0] if "fits_index" in df else "?"
    positions = df.index.to_numpy()
    saturated = (
        df["is_saturated"].to_numpy().astype(bool)
        if "is_saturated" in df.columns
        else np.zeros(len(df), dtype=bool)
    )

    move_positions = df.index[df["sam_z"].diff().abs() > metadata.SAM_Z_BEAM_MOVE]
    if len(move_positions) == 0:
        logger.warning(
            "No direct beam found for scan starting at fits_index %s; not normalizing.",
            scan_id,
        )
        move_position = 0
    else:
        move_position = int(move_positions[0])

    # Direct-beam frames are those strictly before the sample moves into the beam.
    direct = positions < move_position
    usable = direct & ~saturated

    i0, i0_err = 1.0, 0.0
    if usable.any():
        counts = df.loc[usable, "counts_refl"]
        i0 = float(counts.mean())
        i0_err = float(counts.std(ddof=1) / np.sqrt(usable.sum())) if usable.sum() > 1 else 0.0
    elif direct.any():
        logger.warning(
            "All direct-beam frames for scan %s are saturated; I0 is clipped and "
            "R will be under-normalized. Use an attenuated direct-beam measurement.",
            scan_id,
        )
        i0 = float(df.loc[direct, "counts_refl"].mean())

    if i0 == 0 or np.isnan(i0):
        logger.warning("I0 evaluated to %s for scan %s; falling back to 1.0.", i0, scan_id)
        i0, i0_err = 1.0, 0.0

    df["R"] = df["counts_refl"] / i0
    rel_counts = np.where(
        df["counts_refl"] != 0, df["counts_err"] / df["counts_refl"], 0.0
    )
    df["R_err"] = np.abs(df["R"]) * np.sqrt(rel_counts**2 + (i0_err / i0) ** 2)
    df["i0_mask"] = direct.astype(int)
    return df


def mark_stitch_points(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Mark the frames at which a stitch (overlap) boundary begins.

    A boundary is detected between consecutive *reflectivity* frames (direct-beam
    i0 frames are excluded via ``i0_mask``) when either the sample angle steps
    back into already-measured territory (``sam_th`` decreases by more than
    ``config.stitch_theta_backstep``) or a watched condition column
    (``config.stitch_condition_columns``) changes by more than
    ``config.stitch_condition_tol``. The first reflectivity frame is never a
    boundary, so the i0->first-measurement transition — which can step either way
    because of an angle offset — is never marked.

    The motor-settling repeats and the re-measured overlap points that follow a
    boundary are collapsed into a single boundary: once one is marked, no further
    boundary is marked until ``sam_th`` climbs back above the previous segment's
    maximum angle.

    Args:
        df: One scan group (fresh index). Must contain ``sam_th``; ``i0_mask`` and
            the watched condition columns are used when present.
        config: Reduction configuration (back-step and condition thresholds).

    Returns:
        The group with ``mark`` (1 at a boundary, else None) and ``stitch_trigger``
        (text describing what changed, e.g. ``"backstep+exposure"``) columns added.
    """
    df = df.copy()
    n = len(df)
    mark: list[int | None] = [None] * n
    trigger: list[str | None] = [None] * n

    sam_th = df["sam_th"].to_numpy(dtype=float)
    if "i0_mask" in df.columns:
        is_refl = df["i0_mask"].to_numpy() < 1
    else:
        is_refl = np.ones(n, dtype=bool)
    refl = np.where(is_refl)[0]
    if len(refl) < 2:
        df["mark"] = mark
        df["stitch_trigger"] = trigger
        return df

    watched = [c for c in config.stitch_condition_columns if c in df.columns]
    cond_tol = config.stitch_condition_tol + _FLOAT_EPS
    backstep_min = config.stitch_theta_backstep

    prev = int(refl[0])
    running_max = sam_th[prev]
    overlap_until = running_max
    in_overlap = False
    # Compare only consecutive reflectivity frames; the first is never a boundary.
    for raw_pos in refl[1:]:
        pos = int(raw_pos)
        if in_overlap:
            # Stay in the settling/overlap region until we climb past the top of
            # the previous segment (forward measurement has genuinely resumed).
            if sam_th[pos] > overlap_until + _FLOAT_EPS:
                in_overlap = False
        else:
            backstep = sam_th[pos] < sam_th[prev] - backstep_min
            changed = [
                c
                for c in watched
                if abs(float(df[c].iloc[pos]) - float(df[c].iloc[prev])) > cond_tol
            ]
            if backstep or changed:
                reasons = (["backstep"] if backstep else []) + changed
                mark[pos] = 1
                trigger[pos] = "+".join(reasons)
                overlap_until = running_max  # top of the just-finished segment
                in_overlap = True
        running_max = max(running_max, sam_th[pos])
        prev = pos

    df["mark"] = mark
    df["stitch_trigger"] = trigger
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
        logger.info(
            "Stitch at theta %.4f (%s): %d overlap point(s), scale=%.4g +/- %.2g.",
            sam_th.iloc[i],
            df["stitch_trigger"].iloc[i] if "stitch_trigger" in df.columns else "?",
            len(safe_values),
            scale_i,
            scale_err_i,
        )

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


def diagnose_stitches(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Return a per-boundary stitch diagnostic table (no finalize, nothing dropped).

    Runs normalize -> mark -> compute_scale_factors and reports, for every
    detected stitch boundary, what triggered it, the settled before/after values
    of the changed conditions, how many overlap points were used, and the fitted
    scale factor. Use it to see why an expected stitch is missing or mis-scaled.

    Args:
        df: Processed metadata/counts table (as produced by
            :meth:`~pxr_reduce.core.PXRLoader.process`).
        config: Reduction configuration.

    Returns:
        One row per boundary with columns ``scan, fits_index, sam_th, energy,
        polarization, trigger, conditions_changed, num_stitch_points, scale,
        scale_err, failed``. Empty if no boundaries were detected.
    """
    normalized = _apply_per_scan(df, lambda g: normalize_scan(g, config))
    marked = _apply_per_scan(normalized, lambda g: mark_stitch_points(g, config))
    scaled = _apply_per_scan(marked, lambda g: compute_scale_factors(g, config))

    watched = [c for c in config.stitch_condition_columns if c in scaled.columns]
    rows: list[dict[str, Any]] = []
    for scan_id, group in scaled.groupby("scan", sort=True):
        g = group.reset_index(drop=True)
        boundaries = [i for i in range(len(g)) if not pd.isna(g["mark"].iloc[i])]
        for k, b in enumerate(boundaries):
            nxt = boundaries[k + 1] if k + 1 < len(boundaries) else len(g)
            # Settled values: last frame of the previous segment vs last of this
            # one (both are steady measurement frames, past any motor settling).
            changes = []
            for c in watched:
                before = float(g[c].iloc[b - 1])
                after = float(g[c].iloc[nxt - 1])
                if abs(after - before) > config.stitch_condition_tol + _FLOAT_EPS:
                    changes.append(f"{c}: {before:g}->{after:g}")
            rows.append(
                {
                    "scan": scan_id,
                    "fits_index": (
                        int(g["fits_index"].iloc[b]) if "fits_index" in g else b
                    ),
                    "sam_th": float(g["sam_th"].iloc[b]),
                    "energy": float(g["energy"].iloc[b]) if "energy" in g else np.nan,
                    "polarization": (
                        float(g["polarization"].iloc[b])
                        if "polarization" in g
                        else np.nan
                    ),
                    "trigger": g["stitch_trigger"].iloc[b],
                    "conditions_changed": ", ".join(changes),
                    "num_stitch_points": int(g["num_stitch_points"].iloc[b]),
                    "scale": float(g["scale"].iloc[b]),
                    "scale_err": float(g["scale_err"].iloc[b]),
                    "failed": bool(g["failed_stitch_mask"].iloc[b]),
                }
            )
    return pd.DataFrame(rows)
