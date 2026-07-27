"""Sequential beam-spot tracking.

The specular beam moves smoothly across the detector as the angle changes and
weakens at high angle, so a single static mask + global ``argmax`` mislocates it
(it latches onto persistent noise far from the faint spot). This module tracks
the beam frame-to-frame instead:

1. **Local search window** — each frame's beam is sought only within
   ``drift_distance`` of the previous frame's position (:func:`locate_in_window`).
2. **Centroid + SNR** — the position is an intensity-weighted centroid (robust to
   hot pixels), and a peak-to-noise ratio decides whether the beam is actually
   detected; faint frames are flagged as **dropouts** rather than snapping to
   noise.
3. **Trajectory smoothing** — :func:`smooth_track` fits the per-scan path and
   replaces dropouts/outliers with interpolated positions.

All functions are pure and array-based so they can be unit-tested without I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocateResult:
    """Outcome of a windowed beam search.

    Args:
        found: Whether a beam above the SNR threshold was detected.
        y: Row of the beam centroid (or the search center if not found).
        x: Column of the beam centroid (or the search center if not found).
        snr: Peak-to-noise ratio achieved in the window.
    """

    found: bool
    y: int
    x: int
    snr: float


def _local_stats(window: NDArray[np.floating]) -> tuple[float, float]:
    """Return a robust (background, noise) estimate for a window (median, MAD)."""
    background = float(np.median(window))
    mad = float(np.median(np.abs(window - background))) * 1.4826
    if mad <= 0:
        mad = float(window.std()) or 1.0
    return background, mad


def locate_in_window(
    image: NDArray[np.floating],
    center: tuple[float, float],
    radius: int,
    snr_min: float,
    *,
    centroid_frac: float = 0.2,
    centroid_radius: int | None = None,
) -> LocateResult:
    """Locate the beam within ``radius`` pixels of ``center`` in ``image``.

    Background-subtracts the search window, checks the peak-to-noise ratio against
    ``snr_min``, finds the peak, and returns the intensity-weighted centroid of
    the pixels above ``centroid_frac`` of the peak **within ``centroid_radius`` of
    that peak**. Localizing the centroid keeps distant scatter inside a large
    search radius from dragging the position off the beam.

    Args:
        image: The (cleaned) image or sub-region to search.
        center: ``(y, x)`` search center, in ``image`` coordinates.
        radius: Search radius in pixels (how far the beam may have moved).
        snr_min: Minimum peak/noise ratio to count as detected.
        centroid_frac: Only pixels at or above this fraction of the peak signal
            contribute to the centroid.
        centroid_radius: Half-size of the window around the peak used for the
            centroid; defaults to ``radius`` (no localization).

    Returns:
        A :class:`LocateResult`; when not found, ``y``/``x`` are the (rounded)
        search center.
    """
    cy, cx = int(round(center[0])), int(round(center[1]))
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    window = image[y0:y1, x0:x1]
    if window.size == 0:
        return LocateResult(False, cy, cx, 0.0)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    background, noise = _local_stats(window)
    signal = np.clip(window - background, 0.0, None) * circle
    peak = float(signal.max())
    snr = peak / noise if noise > 0 else 0.0
    if peak <= 0 or snr < snr_min:
        return LocateResult(False, cy, cx, snr)

    # Restrict the centroid to a small box around the peak so off-beam scatter
    # elsewhere in the search radius cannot pull the position sideways.
    cr = radius if centroid_radius is None else centroid_radius
    pky, pkx = np.unravel_index(int(np.argmax(signal)), signal.shape)
    ly0, ly1 = max(0, pky - cr), min(signal.shape[0], pky + cr + 1)
    lx0, lx1 = max(0, pkx - cr), min(signal.shape[1], pkx + cr + 1)
    local = signal[ly0:ly1, lx0:lx1]
    lyy = yy[ly0:ly1, lx0:lx1]
    lxx = xx[ly0:ly1, lx0:lx1]

    strong = local >= centroid_frac * peak
    weights = local[strong]
    total = float(weights.sum())
    cyc = float((lyy[strong] * weights).sum() / total)
    cxc = float((lxx[strong] * weights).sum() / total)
    return LocateResult(True, int(round(cyc)), int(round(cxc)), snr)


def anchor_position(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
    *,
    radius: int = 8,
) -> tuple[int, int]:
    """Locate a bright beam globally to seed tracking (e.g. a direct-beam frame).

    Finds the brightest pixel (optionally restricted to ``mask``) and refines it
    to a centroid. Suitable for the intense direct beam, which dominates the frame.

    Args:
        image: The cleaned image.
        mask: Optional region to restrict the initial peak search.
        radius: Refinement window radius around the brightest pixel.

    Returns:
        The ``(y, x)`` anchor position.
    """
    source = image if mask is None else np.where(mask, image, image.min())
    y, x = np.unravel_index(int(np.argmax(source)), image.shape)
    result = locate_in_window(image, (int(y), int(x)), radius, snr_min=0.0)
    return (result.y, result.x)


def smooth_track(
    positions: list[tuple[int, int]],
    found: list[bool],
    *,
    poly_order: int = 3,
    resid_sigma: float = 3.0,
) -> list[tuple[tuple[int, int], bool]]:
    """Fit a smooth per-scan trajectory and correct dropouts/outliers.

    Fits ``y`` and ``x`` versus frame order with a polynomial (using only detected
    frames), rejects residual outliers, and replaces dropouts and outliers with
    the fitted position. Detected inliers keep their measured position.

    Args:
        positions: Per-frame ``(y, x)`` positions in acquisition order.
        found: Per-frame detection flags (True if the beam was detected).
        poly_order: Polynomial order (reduced automatically for short scans).
        resid_sigma: Residual (in robust sigmas) beyond which a detected frame is
            treated as an outlier.

    Returns:
        List of ``((y, x), changed)`` per frame, where ``changed`` is True if the
        position was replaced by the fitted trajectory.
    """
    n = len(positions)
    if n == 0:
        return []
    t = np.arange(n, dtype=float)
    ys = np.array([p[0] for p in positions], dtype=float)
    xs = np.array([p[1] for p in positions], dtype=float)
    good = np.array(found, dtype=bool)

    if good.sum() < 2:
        # Not enough detections to fit; hold detected values, fill the rest.
        logger.warning("Too few beam detections to smooth trajectory; holding.")
        return [(positions[i], not good[i]) for i in range(n)]

    pred_y, in_y = _robust_polyfit(t, ys, good, poly_order, resid_sigma)
    pred_x, in_x = _robust_polyfit(t, xs, good, poly_order, resid_sigma)
    trusted = good & in_y & in_x

    out: list[tuple[tuple[int, int], bool]] = []
    for i in range(n):
        if trusted[i]:
            out.append((positions[i], False))
        else:
            out.append(((int(round(pred_y[i])), int(round(pred_x[i]))), True))
    return out


def _robust_polyfit(
    t: NDArray[np.floating],
    vals: NDArray[np.floating],
    good: NDArray[np.bool_],
    poly_order: int,
    resid_sigma: float,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Fit a polynomial to ``vals[good]`` and return (prediction, inlier mask)."""
    order = min(poly_order, max(1, int(good.sum()) - 1))
    coef = np.polyfit(t[good], vals[good], order)
    pred = np.polyval(coef, t)
    resid = vals - pred
    rstd = float(np.median(np.abs(resid[good]))) * 1.4826
    if rstd <= 0:
        rstd = float(resid[good].std()) or 1.0
    inliers = good & (np.abs(resid) <= resid_sigma * rstd)
    # Refit once without outliers if that leaves enough points.
    if order + 1 <= int(inliers.sum()) < int(good.sum()):
        order = min(order, int(inliers.sum()) - 1)
        coef = np.polyfit(t[inliers], vals[inliers], order)
        pred = np.polyval(coef, t)
    return pred, inliers
