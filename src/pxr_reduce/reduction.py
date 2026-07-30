"""Composable reduction stages: normalize -> mark stitches -> scale -> finalize.

Each stage is a pure function that takes the metadata/counts table plus a
:class:`~pxr_reduce.config.ReductionConfig` and returns an augmented copy. The
top-level :func:`reduce` runs them in order and lets callers skip the stitch/scale
stages for a fast "quick" reduction (see ``apply_scale``).

The stitch-detection and scale-factor algorithms are inherently sequential and
follow the original loader, with these deliberate differences:

* ``print`` diagnostics become ``logging`` calls.
* Per-scan groups are recombined with :func:`pandas.concat` instead of the
  original fragile ``igroup + i`` index arithmetic.
* The stitch scale-factor uncertainty is propagated into ``R_err`` (this was
  commented out in the original, so ``R_err`` previously ignored it).
* Direct-beam (i0) frames can never be used as stitch-overlap points. The
  original matched purely on ``sam_th``, so an i0 frame whose angle recurred
  after a boundary was averaged into the pre-change overlap at ``R ~= 1`` and
  corrupted the fitted scale.
* A boundary that cannot be fitted no longer aborts the scan. Every later
  boundary is still evaluated and reported, and the failure is recorded per
  boundary (``stitch_failed``, ``stitch_fail_reason``) as well as per frame
  (``failed_stitch_mask``).
* Degenerate scale factors (non-finite, zero, negative) are rejected by the fit
  and surfaced as failed stitches instead of silently zeroing or sign-flipping
  ``R``.
* Every fitted stitch is quality-checked (``stitch_suspect``,
  ``stitch_quality_note``): the overlap points must agree about the scale, and
  where the scale is predictable from the condition change it must match. A
  suspect stitch is reported and its scale still applied — the check exists to
  catch a fit that succeeded but is wrong, which the original had no way to see.

Expected input columns: ``scan``, ``sam_th``, ``sam_z``, ``q``, ``energy``,
``polarization``, ``counts_refl``, ``counts_err``, ``counts_ratio``,
``is_saturated``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from pxr_reduce import metadata
from pxr_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

# Tolerance for comparing pre-rounded metadata values (angles, conditions).
_FLOAT_EPS = 1e-9

# Granularity at which duplicate points may be averaged together.
_DUPLICATE_SCOPES: frozenset[str] = frozenset({"sweep", "scan", "angle"})


class StitchFitError(RuntimeError):
    """Raised when a stitch scale factor cannot be reliably determined.

    Carries a short human-readable reason, which is recorded in the
    ``stitch_fail_reason`` column and reported by :func:`diagnose_stitches`.
    """


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

    A physical stitch factor is finite and strictly positive, so degenerate
    results are rejected rather than returned. Letting a zero or negative scale
    through would silently zero or sign-flip ``R`` for the rest of the scan, and
    those points would then vanish in :func:`finalize`'s ``R > 0`` filter with no
    indication of why.

    Args:
        pre_r: Pre-change reflectivity values.
        post_r: Post-change reflectivity values at the same angles.

    Returns:
        A ``(scale, scale_err)`` tuple.

    Raises:
        StitchFitError: If the overlap is empty, either side is non-finite, the
            pre-change values are all zero, the fit does not converge, or the
            fitted scale is non-finite or non-positive.
    """
    if len(pre_r) == 0 or len(post_r) == 0:
        raise StitchFitError("no overlapping stitch points")
    if not (np.all(np.isfinite(pre_r)) and np.all(np.isfinite(post_r))):
        raise StitchFitError("overlap reflectivity is not finite")
    if not np.any(pre_r != 0.0):
        raise StitchFitError("all pre-change overlap reflectivities are zero")

    try:
        with warnings.catch_warnings():
            # A single overlap point leaves no degrees of freedom, so scipy warns
            # that the covariance is inestimable. That case is detected below and
            # reported as a suspect stitch with a clearer message, so the raw
            # warning is redundant noise during a batch reduction.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(stitch_ratio_model, pre_r, post_r, p0=[1.0])
    except (RuntimeError, ValueError) as e:
        raise StitchFitError(f"scale fit did not converge ({e})") from e

    scale = float(popt[0])
    if not np.isfinite(scale):
        raise StitchFitError("fitted scale is not finite")
    if scale <= 0.0:
        raise StitchFitError(f"fitted scale is non-positive ({scale:.4g})")

    variance = float(pcov[0, 0])
    estimable = np.isfinite(variance) and variance >= 0.0
    scale_err = float(np.sqrt(variance)) if estimable else 0.0
    return scale, scale_err


def _overlap_rms_rel(
    pre_r: np.ndarray, post_r: np.ndarray, scale: float
) -> float:
    """Relative RMS disagreement of the overlap points about the fitted scale.

    A well-matched stitch has every overlap angle agreeing on one scale factor, so
    this is near zero; a large value means the two segments do not describe the same
    curve where they overlap (mismatched angles, drifting beam, contaminated points)
    even though the fit still returned a number.

    Args:
        pre_r: Pre-change reflectivity at the overlap angles.
        post_r: Post-change reflectivity at the same angles, in the same order.
        scale: The fitted scale factor.

    Returns:
        The relative RMS residual, or NaN when it cannot be evaluated. A single
        overlap point is fitted exactly and would give a misleading 0.0, so it
        returns NaN too.
    """
    if len(pre_r) < 2:
        return float("nan")
    model = scale * pre_r
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(model != 0.0, (post_r - model) / model, np.nan)
    if not np.all(np.isfinite(rel)):
        return float("nan")
    return float(np.sqrt(np.mean(rel**2)))


def _expected_scale(
    trigger: str | None, config: ReductionConfig
) -> tuple[float | None, str]:
    """Return the scale a boundary should fit, when that is predictable, and why.

    Reflectivity is already normalized by exposure and beam current, so a boundary
    triggered only by conditions in ``config.stitch_normalized_conditions`` — or by
    a bare ``sam_th`` back-step, where nothing the reduction watches changed at all —
    must fit ~1.0. Boundaries that change the incident flux (slits, higher-order
    suppressor) legitimately fit something else and are not predictable.

    The bare-back-step case is the most informative: a scale far from 1.0 there means
    either the beam/sample drifted between the two passes, or something did change
    that is absent from ``config.stitch_condition_columns``.

    Args:
        trigger: The ``stitch_trigger`` text, e.g. ``"backstep+exposure"``.
        config: Reduction configuration.

    Returns:
        An ``(expected, basis)`` tuple; ``expected`` is None when no prediction can
        be made, and ``basis`` describes the reasoning for a diagnostic message.
    """
    changed = [c for c in str(trigger or "").split("+") if c and c != "backstep"]
    if not changed:
        return 1.0, "no watched condition changed here"
    if all(c in config.stitch_normalized_conditions for c in changed):
        return 1.0, f"{', '.join(changed)} is already normalized out"
    return None, ""


def _assess_stitch(
    scale: float,
    n_points: int,
    rms_rel: float,
    expected: float | None,
    basis: str,
    config: ReductionConfig,
) -> tuple[int, str]:
    """Judge a fitted stitch and describe anything questionable about it.

    Purely diagnostic: a suspect stitch is reported, never dropped. The point is to
    surface a scale that fitted successfully but is probably wrong, which is the
    failure mode that silently corrupts a reduction.

    Args:
        scale: The fitted scale factor.
        n_points: Number of overlap angles used.
        rms_rel: Relative RMS residual from :func:`_overlap_rms_rel`.
        expected: Expected scale from :func:`_expected_scale`, or None.
        basis: Why that scale is expected, for the reported note.
        config: Reduction configuration (the two quality thresholds).

    Returns:
        A ``(suspect, note)`` tuple; ``note`` is empty when nothing is wrong.
    """
    notes: list[str] = []
    if n_points == 1:
        notes.append("only one overlap point, so the scale is unverifiable")
    elif np.isfinite(rms_rel) and rms_rel > config.stitch_max_overlap_rms:
        notes.append(f"overlap points disagree by {rms_rel:.1%} rms")

    if expected is not None and expected != 0.0:
        deviation = abs(scale / expected - 1.0)
        if deviation > config.stitch_max_scale_deviation:
            notes.append(
                f"scale {scale:.4g} is {deviation:.1%} from the expected "
                f"{expected:.4g} ({basis})"
            )
    return (1 if notes else 0), "; ".join(notes)


def _init_stitch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stitch bookkeeping columns with their neutral defaults.

    The float quality columns default to NaN rather than 0.0 so an unevaluated
    boundary is never mistaken for a perfectly-agreeing one.

    Args:
        df: Table to annotate (modified in place).

    Returns:
        The same table, for chaining.
    """
    df["scale"] = 1.0
    df["scale_err"] = 0.0
    df["num_stitch_points"] = 0
    df["failed_stitch_mask"] = 0
    df["stitch_failed"] = 0
    df["stitch_fail_reason"] = None
    df["overlap_rms_rel"] = np.nan
    df["expected_scale"] = np.nan
    df["stitch_suspect"] = 0
    df["stitch_quality_note"] = None
    return df


def _record_failed_stitch(df: pd.DataFrame, index: int, reason: str) -> None:
    """Flag a boundary whose scale could not be determined and mask downstream.

    Marks the boundary itself in ``stitch_failed``/``stitch_fail_reason``, then
    sets ``failed_stitch_mask`` from the boundary to the end of the scan: later
    segments can still be stitched to each other, but their absolute level is tied
    to this boundary's unknown factor, so their normalization is unestablished.

    Args:
        df: The scan group being scaled (fresh index; modified in place).
        index: Positional index of the failed boundary.
        reason: Short explanation, recorded and logged.
    """
    logger.warning(
        "Failed stitch at index %d (energy %s eV, theta %s): %s. Absolute scale is "
        "unestablished from here to the end of the scan.",
        index,
        df["energy"].iloc[index] if "energy" in df.columns else "?",
        df["sam_th"].iloc[index],
        reason,
    )
    df.loc[index, "stitch_failed"] = 1
    df.loc[index, "stitch_fail_reason"] = reason
    df.loc[index:, "failed_stitch_mask"] = 1


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

    Two distinct sets of frames are marked, because they are not the same set:

    ``is_direct_beam``
        Sample out of the beam. These and only these form the I0 average.
    ``i0_mask``
        Not a reflectivity measurement — the direct-beam frames *plus* any
        **transition** frames, where ``sam_z`` has already moved the sample in but
        ``sam_th`` has not started sweeping. Such a frame measures neither the direct
        beam nor a reflection, so it must be excluded from the output and from stitch
        overlaps; folding it into the I0 average instead would bias I0, since the
        sample is already blocking the beam.

    The transition run is found from the *nominal* angle: it is the frames at or after
    the ``sam_z`` move whose ``sam_th`` still matches the direct-beam angle, within
    ``config.stitch_theta_backstep`` (the smallest angle change the reduction treats as
    real).

    Args:
        df: One scan group (fresh index).
        config: Reduction configuration (angle tolerance for the transition run).

    Returns:
        The group with ``R``, ``R_err``, ``i0_mask``, and ``is_direct_beam`` columns
        added.
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
    transition = _transition_frames(df, move_position, config)
    if transition.any():
        logger.info(
            "Scan starting at fits_index %s: %d frame(s) have the sample in the beam "
            "at the un-swept angle; excluded from both I0 and the output.",
            scan_id,
            int(transition.sum()),
        )
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
    df["is_direct_beam"] = direct.astype(int)
    df["i0_mask"] = (direct | transition).astype(int)
    return df


def _transition_frames(
    df: pd.DataFrame, move_position: int, config: ReductionConfig
) -> np.ndarray:
    """Flag frames where the sample is in the beam but the angle has not moved yet.

    ``sam_z`` moves the sample in one frame before the sweep starts, so the frame at
    the move still sits at the direct-beam angle. It measures neither I0 nor a
    reflection, and left alone it becomes a spurious output point at q ~= 0.

    Args:
        df: One scan group (fresh index).
        move_position: Positional index of the ``sam_z`` move.
        config: Reduction configuration (angle tolerance).

    Returns:
        Boolean mask over the group, True for the transition run.
    """
    flags = np.zeros(len(df), dtype=bool)
    if move_position <= 0 or "sam_th" not in df.columns:
        return flags

    sam_th = df["sam_th"].to_numpy(dtype=float)
    # The angle held while the sample was out of the beam.
    direct_angle = sam_th[move_position - 1]
    tolerance = config.stitch_theta_backstep
    for position in range(move_position, len(df)):
        if abs(sam_th[position] - direct_angle) > tolerance:
            break
        flags[position] = True
    return flags


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


@dataclass(frozen=True)
class OverlapPoint:
    """One candidate stitch-overlap frame and what became of it.

    Args:
        index: Positional index of the frame within its scan group.
        side: ``"pre"`` or ``"post"`` — which side of the boundary it sits on.
        sam_th: The frame's sample angle.
        used: Whether it contributed to the fitted scale factor.
        reason: Why it was excluded; empty when ``used`` is True.
    """

    index: int
    side: str
    sam_th: float
    used: bool
    reason: str


@dataclass(frozen=True)
class OverlapSelection:
    """The overlap points chosen at one stitch boundary, with every rejection.

    Args:
        boundary: Positional index of the boundary frame.
        repeat: Number of consecutive frames at the boundary angle (the settling
            repeats, excluded from the post-change window).
        angles: The matched angles actually fitted, ascending.
        pre_r: Pre-change reflectivity at ``angles`` (repeats averaged).
        post_r: Post-change reflectivity at ``angles`` (repeats averaged).
        points: Every candidate frame considered, used or rejected.
        n_unmatched: Frames in the pre-change segment whose angle was never
            re-measured after the boundary. These were never overlap candidates, so
            they are counted rather than enumerated.
    """

    boundary: int
    repeat: int
    angles: list[float]
    pre_r: np.ndarray
    post_r: np.ndarray
    points: list[OverlapPoint]
    n_unmatched: int

    def dropped(self) -> list[OverlapPoint]:
        """Return only the rejected candidates."""
        return [p for p in self.points if not p.used]


def _select_overlap(
    df: pd.DataFrame,
    boundary: int,
    prev_mark_index: int,
    is_refl: np.ndarray,
    config: ReductionConfig,
) -> OverlapSelection:
    """Match the pre/post overlap points at one stitch boundary.

    This is the single implementation of overlap selection: :func:`compute_scale_factors`
    fits what it returns, and the stitch diagnostics report what it rejected. Keeping
    them on one code path is the point — a report derived from a second, parallel
    implementation would eventually describe a selection that never happened.

    Angles are matched by exact equality on the pre-rounded ``sam_th``. Candidates are
    rejected, in this precedence order, for being a direct-beam frame, being saturated,
    or falling at or below ``config.stitch_cutoff`` in spot/dark ratio. An angle
    surviving on only one side cannot be fitted, and its partner is reported as such.

    Args:
        df: One scan group (fresh index), marked and normalized.
        boundary: Positional index of the boundary frame.
        prev_mark_index: Index of the previous boundary (start of the pre segment).
        is_refl: Boolean array, True where a frame is a reflectivity (non-i0) frame.
        config: Reduction configuration.

    Returns:
        The :class:`OverlapSelection` for this boundary.
    """
    sam_th = df["sam_th"]
    i = boundary
    ratio_cut = config.stitch_cutoff
    points: list[OverlapPoint] = []

    # Consecutive repeated angles at the boundary: motor settling, not overlap.
    sam_th_stitch = sam_th.iloc[i]
    repeat = 0
    for val in sam_th.iloc[i:]:
        if val == sam_th_stitch:
            repeat += 1
        else:
            break

    post_values = sam_th.iloc[i + repeat :].values
    n_i0_excluded = 0
    n_unmatched = 0

    # Pre-change candidates: angles that recur after the change.
    ipre: list[int] = []
    for j, val in enumerate(sam_th.iloc[prev_mark_index:i]):
        jj = j + prev_mark_index
        if val not in post_values:
            if is_refl[jj]:
                n_unmatched += 1
            continue
        if jj in ipre:
            continue
        if not is_refl[jj]:
            n_i0_excluded += 1
            points.append(
                OverlapPoint(jj, "pre", float(val), False, "direct-beam (i0) frame")
            )
        elif df["is_saturated"].iloc[jj]:
            logger.info("Saturated pre-change stitch point dropped: %d", jj)
            points.append(OverlapPoint(jj, "pre", float(val), False, "saturated"))
        elif not df["counts_ratio"].iloc[jj] > ratio_cut:
            points.append(
                OverlapPoint(
                    jj, "pre", float(val), False,
                    f"counts_ratio {df['counts_ratio'].iloc[jj]:.4g} <= "
                    f"stitch_cutoff {ratio_cut:g}",
                )
            )
        else:
            ipre.append(jj)

    # Post-change candidates: angles still present on the surviving pre side. A post
    # frame whose pre partner was already rejected never becomes a candidate here,
    # so it is reported against the raw pre angles below.
    pre_values = sam_th.iloc[ipre].values
    raw_pre_values = sam_th.iloc[prev_mark_index:i].values
    ipost: list[int] = []
    for j, val in enumerate(sam_th.iloc[i + repeat :]):
        jj = j + i + repeat
        if not pd.isna(df["mark"].iloc[jj]):
            break
        if val not in pre_values:
            if val in raw_pre_values and is_refl[jj]:
                points.append(
                    OverlapPoint(
                        jj, "post", float(val), False,
                        "partner dropped (no surviving pre-change point at this angle)",
                    )
                )
            continue
        if jj in ipost:
            continue
        if not is_refl[jj]:
            n_i0_excluded += 1
            points.append(
                OverlapPoint(jj, "post", float(val), False, "direct-beam (i0) frame")
            )
        elif df["is_saturated"].iloc[jj]:
            logger.info("Saturated post-change stitch point dropped: %d", jj)
            points.append(OverlapPoint(jj, "post", float(val), False, "saturated"))
        elif not df["counts_ratio"].iloc[jj] > ratio_cut:
            points.append(
                OverlapPoint(
                    jj, "post", float(val), False,
                    f"counts_ratio {df['counts_ratio'].iloc[jj]:.4g} <= "
                    f"stitch_cutoff {ratio_cut:g}",
                )
            )
        else:
            ipost.append(jj)

    if n_i0_excluded:
        logger.info(
            "Excluded %d direct-beam (i0) frame(s) from the overlap at index %d; "
            "their angle recurs across the boundary.",
            n_i0_excluded,
            i,
        )

    stitch_pre = df[["sam_th", "R"]].iloc[ipre]
    stitch_post = df[["sam_th", "R"]].iloc[ipost]
    angles = sorted(set(stitch_pre["sam_th"]) & set(stitch_post["sam_th"]))

    # A surviving frame whose angle is missing on the other side cannot be paired.
    for indices, side in ((ipre, "pre"), (ipost, "post")):
        for jj in indices:
            angle = float(sam_th.iloc[jj])
            paired = angle in angles
            other = "post" if side == "pre" else "pre"
            points.append(
                OverlapPoint(
                    jj, side, angle, paired,
                    "" if paired
                    else f"partner dropped (no surviving {other}-change point "
                         "at this angle)",
                )
            )

    pre_r = stitch_pre.groupby("sam_th")["R"].mean().loc[angles].to_numpy()
    post_r = stitch_post.groupby("sam_th")["R"].mean().loc[angles].to_numpy()
    return OverlapSelection(
        boundary=i,
        repeat=repeat,
        angles=[float(a) for a in angles],
        pre_r=pre_r,
        post_r=post_r,
        points=sorted(points, key=lambda p: p.index),
        n_unmatched=n_unmatched,
    )


def _iter_overlap_selections(
    df: pd.DataFrame, config: ReductionConfig
) -> Iterator[tuple[int, OverlapSelection]]:
    """Yield ``(boundary_index, selection)`` for every marked boundary in a scan.

    Owns the ``prev_mark_index`` bookkeeping so the fit and the diagnostics walk the
    boundaries identically. The pre-change segment for a boundary always starts at the
    previous boundary, whether or not that one could be fitted.

    Args:
        df: One scan group (fresh index), marked and normalized.
        config: Reduction configuration.

    Yields:
        The boundary's positional index and its overlap selection.
    """
    # Direct-beam frames are measured with the sample out of the beam (R ~= 1), so
    # they are never valid overlap points even when their angle recurs.
    if "i0_mask" in df.columns:
        is_refl = df["i0_mask"].to_numpy() < 1
    else:
        is_refl = np.ones(len(df), dtype=bool)

    prev_mark_index = 0
    for i in range(len(df)):
        if pd.isna(df["mark"].iloc[i]):
            continue
        yield i, _select_overlap(df, i, prev_mark_index, is_refl, config)
        prev_mark_index = i


def compute_scale_factors(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Compute cumulative stitch scale factors for one scan.

    At each marked boundary, matches overlapping ``sam_th`` points before and
    after the change (excluding direct-beam, saturated, and low-SNR frames), fits
    the ratio of their reflectivities with a one-parameter least-squares fit, and
    accumulates the scale factor and its uncertainty from that point onward.

    A boundary that cannot be fitted does not stop the scan: the failure is
    recorded, every frame from it onward is flagged as having an unestablished
    absolute scale, and the remaining boundaries are still matched and fitted
    against their own predecessors. That keeps the diagnostics complete (each
    boundary reports its own overlap count and outcome) without publishing a tail
    whose normalization is unknown — :func:`finalize` still drops it when
    ``config.drop_failed_stitch`` is set.

    Args:
        df: One scan group (fresh index) that has been marked and normalized.
        config: Reduction configuration (stitch cutoff, drop-failed policy).

    Returns:
        The group with ``scale``, ``scale_err``, ``num_stitch_points``,
        ``failed_stitch_mask``, ``stitch_failed``, and ``stitch_fail_reason``
        columns added.
    """
    df = _init_stitch_columns(df.copy())
    sam_th = df["sam_th"]

    for i, sel in _iter_overlap_selections(df, config):
        df.loc[i, "num_stitch_points"] = len(sel.angles)

        if not sel.angles:
            _record_failed_stitch(df, i, "no overlapping stitch points")
            continue

        try:
            scale_i, scale_err_i = _fit_scale_factor(sel.pre_r, sel.post_r)
        except StitchFitError as e:
            _record_failed_stitch(df, i, str(e))
            continue

        trigger_i = (
            df["stitch_trigger"].iloc[i] if "stitch_trigger" in df.columns else None
        )
        rms_rel = _overlap_rms_rel(sel.pre_r, sel.post_r, scale_i)
        expected, basis = _expected_scale(trigger_i, config)
        suspect, note = _assess_stitch(
            scale_i, len(sel.angles), rms_rel, expected, basis, config
        )
        df.loc[i, "overlap_rms_rel"] = rms_rel
        df.loc[i, "expected_scale"] = np.nan if expected is None else expected
        df.loc[i, "stitch_suspect"] = suspect
        df.loc[i, "stitch_quality_note"] = note or None

        logger.info(
            "Stitch at theta %.4f (%s): %d overlap point(s), scale=%.4g +/- %.2g.",
            sam_th.iloc[i],
            trigger_i if trigger_i is not None else "?",
            len(sel.angles),
            scale_i,
            scale_err_i,
        )
        if suspect:
            logger.warning(
                "Suspect stitch at index %d (energy %s eV, theta %.4f, trigger %s): "
                "%s. The scale was applied; review before using these points.",
                i,
                df["energy"].iloc[i] if "energy" in df.columns else "?",
                sam_th.iloc[i],
                trigger_i if trigger_i is not None else "?",
                note,
            )

        prev_scale = df["scale"].iloc[i:].to_numpy()
        prev_scale_err = df["scale_err"].iloc[i:].to_numpy()
        new_scale = prev_scale * scale_i
        rel_prev = np.where(prev_scale != 0, prev_scale_err / prev_scale, 0.0)
        rel_i = scale_err_i / scale_i if scale_i != 0 else 0.0
        new_scale_err = np.abs(new_scale) * np.sqrt(rel_prev**2 + rel_i**2)
        df.loc[i:, "scale"] = new_scale
        df.loc[i:, "scale_err"] = new_scale_err

    return df


def apply_scaling(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Divide ``R`` by the cumulative scale factor and propagate its error.

    This is the step whose error propagation was commented out in the original
    loader; ``R_err`` now includes the stitch scale-factor uncertainty.

    A non-finite or non-positive scale yields ``NaN`` rather than the original's
    ``0.0``: a zero is a plausible-looking reflectivity that disappears silently in
    :func:`finalize`, whereas ``NaN`` is unmistakable. :func:`_fit_scale_factor`
    already rejects such factors, so reaching this guard indicates a scale that
    came from somewhere else.

    Args:
        df: Table with ``R``, ``R_err``, ``scale``, ``scale_err`` columns.
        config: Reduction configuration (unused; kept for signature consistency).

    Returns:
        The table with scaled ``R`` and ``R_err``.
    """
    df = df.copy()
    scale = df["scale"].to_numpy(dtype=float)
    scale_err = df["scale_err"].to_numpy(dtype=float)
    r = df["R"].to_numpy(dtype=float)
    r_err = df["R_err"].to_numpy(dtype=float)

    usable = np.isfinite(scale) & (scale > 0.0)
    if not usable.all():
        logger.error(
            "%d point(s) have a non-positive or non-finite stitch scale factor; "
            "their reflectivity is set to NaN rather than silently zeroed.",
            int((~usable).sum()),
        )

    nan_fill = np.full(r.shape, np.nan)
    r_scaled = np.divide(r, scale, out=nan_fill.copy(), where=usable)
    rel_r = np.divide(r_err, r, out=np.zeros(r.shape), where=r != 0.0)
    rel_scale = np.divide(scale_err, scale, out=np.zeros(scale.shape), where=usable)
    df["R"] = r_scaled
    df["R_err"] = np.abs(r_scaled) * np.sqrt(rel_r**2 + rel_scale**2)
    return df


def finalize(
    df: pd.DataFrame,
    config: ReductionConfig,
    drop_duplicates: bool = True,
    duplicate_scope: str = "sweep",
) -> pd.DataFrame:
    """Select valid reduced points and optionally average duplicates.

    Drops direct-beam frames, saturated frames, non-positive reflectivity, and
    (when configured) failed-stitch points.

    ``scan_id`` and ``sweep`` are carried through and, by default, are part of the
    duplicate key, so every sweep is exported as its own profile. That is the safe
    default: two sweeps that happen to share an energy and polarization may be
    deliberate repeats or an accidental duplicate, and nothing can tell them apart
    automatically — silently averaging them corrupts the result invisibly, while
    keeping them apart merely yields two visible curves. ``duplicate_scope`` relaxes
    this when repeats *should* be combined.

    Overlap points re-measured *within* one sweep are always averaged, whatever the
    scope, since those are the same measurement.

    Args:
        df: Fully reduced table.
        config: Reduction configuration (drop-failed policy).
        drop_duplicates: If True, average points sharing the duplicate key.
        duplicate_scope: Granularity of that key — ``"sweep"`` (default) keeps every
            sweep separate, ``"scan"`` merges repeat sweeps within a scan, ``"angle"``
            merges everything sharing (sam_th, energy, polarization).

    Returns:
        Reduced dataset with columns ``scan_id, sweep, energy, polarization, sam_th, q,
        R, R_err``; identifier columns absent from the input are omitted.

    Raises:
        ValueError: If ``duplicate_scope`` is not one of the three known values.
    """
    if duplicate_scope not in _DUPLICATE_SCOPES:
        raise ValueError(
            f"duplicate_scope must be one of {sorted(_DUPLICATE_SCOPES)}, "
            f"got {duplicate_scope!r}"
        )

    mask = df["i0_mask"] < 1
    mask &= ~df["is_saturated"].astype(bool)
    mask &= df["R"] > 0
    if config.drop_failed_stitch:
        mask &= df["failed_stitch_mask"] < 1

    identifiers = [c for c in ("scan_id", "sweep") if c in df.columns]
    keys = ["sam_th", "energy", "polarization"]
    if duplicate_scope == "sweep":
        keys = identifiers + keys
    elif duplicate_scope == "scan":
        keys = [c for c in identifiers if c == "scan_id"] + keys

    columns = identifiers + ["energy", "polarization", "sam_th", "q", "R", "R_err"]
    out = df.loc[mask, columns]
    if drop_duplicates:
        # Identifiers not in the key would otherwise be averaged into nonsense.
        out = out.groupby(keys, as_index=False).mean()
        out = out[[c for c in columns if c in out.columns]]
    return out.reset_index(drop=True)


def reduce(
    df: pd.DataFrame,
    config: ReductionConfig,
    *,
    apply_scale: bool = True,
    drop_duplicates: bool = True,
    duplicate_scope: str = "sweep",
) -> pd.DataFrame:
    """Run the full reduction pipeline.

    Args:
        df: Metadata/counts table (see module docstring for required columns).
        config: Reduction configuration.
        apply_scale: If False, skip stitch detection and scaling (quick mode) —
            reflectivity is I0-normalized only. Useful for fast previews that
            avoid stitch-overlap pitfalls.
        drop_duplicates: Average points sharing the duplicate key.
        duplicate_scope: Granularity of that key; see :func:`finalize`.

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
        df = _init_stitch_columns(df.copy())

    return finalize(
        df,
        config,
        drop_duplicates=drop_duplicates,
        duplicate_scope=duplicate_scope,
    )


def annotate(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Run normalize -> mark -> compute_scale_factors, without finalizing.

    Returns the per-frame table augmented with the reduction's working columns
    (``R``, ``R_err``, ``i0_mask``, ``mark``, ``stitch_trigger``, ``scale``,
    ``scale_err``, ``num_stitch_points``, ``failed_stitch_mask``, ``stitch_failed``,
    ``stitch_fail_reason``, ``overlap_rms_rel``, ``expected_scale``,
    ``stitch_suspect``, ``stitch_quality_note``) but with no rows dropped or
    averaged. Used by diagnostics/plots that need per-frame stitch and scale
    information alongside the raw counts.

    Args:
        df: Processed metadata/counts table.
        config: Reduction configuration.

    Returns:
        The annotated per-frame table.
    """
    df = _apply_per_scan(df, lambda g: normalize_scan(g, config))
    df = _apply_per_scan(df, lambda g: mark_stitch_points(g, config))
    return _apply_per_scan(df, lambda g: compute_scale_factors(g, config))


def overlap_report(
    df: pd.DataFrame, config: ReductionConfig, *, annotated: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return every stitch-overlap candidate point and whether it was used.

    Reports the *actual* selection made by :func:`compute_scale_factors` — both share
    :func:`_select_overlap` — so this is the authoritative answer to "why was that
    point not used in the stitch?".

    Args:
        df: Processed metadata/counts table (as produced by
            :meth:`~pxr_reduce.core.PXRLoader.process`).
        config: Reduction configuration.
        annotated: Pre-computed :func:`annotate` output, to avoid re-running the
            reduction stages when several reports are built from one table.

    Returns:
        One row per candidate frame per boundary, with columns ``scan,
        boundary_index, boundary_sam_th, energy, polarization, fits_index, side,
        sam_th, R, counts_ratio, is_saturated, n_sat_roi, n_sat_dark, used, reason``.
        Empty if no boundaries were detected.
    """
    scaled = annotate(df, config) if annotated is None else annotated
    rows: list[dict[str, Any]] = []
    for scan_id, group in scaled.groupby("scan", sort=True):
        g = group.reset_index(drop=True)
        for i, sel in _iter_overlap_selections(g, config):
            for point in sel.points:
                j = point.index
                rows.append(
                    {
                        "scan": scan_id,
                        "boundary_index": i,
                        "boundary_sam_th": float(g["sam_th"].iloc[i]),
                        "energy": (
                            float(g["energy"].iloc[j]) if "energy" in g else np.nan
                        ),
                        "polarization": (
                            float(g["polarization"].iloc[j])
                            if "polarization" in g
                            else np.nan
                        ),
                        "fits_index": (
                            int(g["fits_index"].iloc[j]) if "fits_index" in g else j
                        ),
                        "side": point.side,
                        "sam_th": point.sam_th,
                        "R": float(g["R"].iloc[j]) if "R" in g else np.nan,
                        "counts_ratio": (
                            float(g["counts_ratio"].iloc[j])
                            if "counts_ratio" in g
                            else np.nan
                        ),
                        "is_saturated": bool(g["is_saturated"].iloc[j]),
                        "n_sat_roi": (
                            int(g["n_sat_roi"].iloc[j]) if "n_sat_roi" in g else 0
                        ),
                        "n_sat_dark": (
                            int(g["n_sat_dark"].iloc[j]) if "n_sat_dark" in g else 0
                        ),
                        "used": point.used,
                        "reason": point.reason,
                    }
                )
    return pd.DataFrame(rows)


def summarize_stitches(report: pd.DataFrame | None) -> dict[str, int]:
    """Count stitch boundaries by outcome.

    Args:
        report: A :func:`diagnose_stitches` table, or None when no scaling ran.

    Returns:
        Counts keyed ``total``, ``ok``, ``suspect``, ``failed``. A failed boundary
        is never also counted as suspect, so the three add up to ``total``.
    """
    if report is None or not len(report):
        return {"total": 0, "ok": 0, "suspect": 0, "failed": 0}
    failed = int(report["failed"].sum())
    suspect = int((report["suspect"] & ~report["failed"]).sum())
    return {
        "total": len(report),
        "ok": len(report) - failed - suspect,
        "suspect": suspect,
        "failed": failed,
    }


def diagnose_stitches(
    df: pd.DataFrame, config: ReductionConfig, *, annotated: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return a per-boundary stitch diagnostic table (no finalize, nothing dropped).

    Runs :func:`annotate` and reports, for every detected stitch boundary, what
    triggered it, the settled before/after values of the changed conditions, how
    many overlap points were used, the fitted scale factor, and whether that scale
    survives its quality checks. Use it to see why an expected stitch is missing or
    mis-scaled.

    Args:
        df: Processed metadata/counts table (as produced by
            :meth:`~pxr_reduce.core.PXRLoader.process`).
        config: Reduction configuration.
        annotated: Pre-computed :func:`annotate` output, to avoid re-running the
            reduction stages when several reports are built from one table.

    Returns:
        One row per boundary with columns ``scan, boundary_index, fits_index, sam_th,
        energy, polarization, trigger, conditions_changed, num_stitch_points, scale,
        scale_err, overlap_rms_rel, expected_scale, failed, fail_reason, suspect,
        quality_note, scale_established``. ``failed`` refers to that boundary's own
        fit; ``suspect`` means it fitted but did not survive its quality checks (see
        ``quality_note``) and the scale was still applied; ``scale_established`` is
        False whenever this or any earlier boundary in the same scan failed — those
        points are internally stitched but sit at an unknown absolute level. Empty
        if no boundaries were detected.
    """
    scaled = annotate(df, config) if annotated is None else annotated

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
            reason = g["stitch_fail_reason"].iloc[b]
            note = g["stitch_quality_note"].iloc[b]
            rows.append(
                {
                    "scan": scan_id,
                    # Stable per-scan identifiers, used to name diagnostic outputs.
                    "scan_id": (
                        int(g["scan_id"].iloc[b]) if "scan_id" in g else -1
                    ),
                    "sweep": int(g["sweep"].iloc[b]) if "sweep" in g else 0,
                    # Positional index within the scan group; joins this table to
                    # :func:`overlap_report`.
                    "boundary_index": b,
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
                    "overlap_rms_rel": float(g["overlap_rms_rel"].iloc[b]),
                    "expected_scale": float(g["expected_scale"].iloc[b]),
                    "failed": bool(g["stitch_failed"].iloc[b]),
                    "fail_reason": reason if isinstance(reason, str) else "",
                    "suspect": bool(g["stitch_suspect"].iloc[b]),
                    "quality_note": note if isinstance(note, str) else "",
                    "scale_established": not bool(g["failed_stitch_mask"].iloc[b]),
                }
            )
    return pd.DataFrame(rows)
