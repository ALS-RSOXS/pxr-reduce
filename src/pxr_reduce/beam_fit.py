"""Beam-shape estimation for data-driven ROI sizing.

Estimates the direct-beam (i0) footprint from image moments and turns the
aggregated shape into a rectangular ROI. Moments give a fast, robust Gaussian
sigma without a nonlinear fit; a full 2D-Gaussian fit can be added later behind
a config switch, and per-energy aggregation on top of :func:`aggregate_shapes`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BeamShape:
    """Estimated beam centroid and Gaussian widths (all in pixels).

    Args:
        centroid_y: Intensity-weighted row centroid.
        centroid_x: Intensity-weighted column centroid.
        sigma_y: Gaussian sigma along rows.
        sigma_x: Gaussian sigma along columns.
        amplitude: Peak value in the fit window.
        success: Whether the estimate produced finite, positive sigmas.
    """

    centroid_y: float
    centroid_x: float
    sigma_y: float
    sigma_x: float
    amplitude: float
    success: bool


def estimate_moments(window: NDArray[np.floating]) -> BeamShape:
    """Estimate beam centroid and sigmas from background-subtracted moments.

    The window median is subtracted as a background estimate and negatives are
    clipped so background pixels do not inflate the widths.

    Args:
        window: Small image window centred on the beam, in ADU.

    Returns:
        A :class:`BeamShape`; ``success`` is False if the window has no signal.
    """
    data = np.asarray(window, dtype=float)
    peak = float(data.max()) if data.size else 0.0
    background = float(np.median(data)) if data.size else 0.0
    weights = np.clip(data - background, 0.0, None)
    total = weights.sum()
    if data.size == 0 or total <= 0:
        return BeamShape(np.nan, np.nan, np.nan, np.nan, peak, False)

    yy, xx = np.indices(data.shape)
    cy = float((yy * weights).sum() / total)
    cx = float((xx * weights).sum() / total)
    var_y = float((weights * (yy - cy) ** 2).sum() / total)
    var_x = float((weights * (xx - cx) ** 2).sum() / total)
    sigma_y = float(np.sqrt(var_y))
    sigma_x = float(np.sqrt(var_x))
    success = bool(
        np.isfinite(sigma_y) and np.isfinite(sigma_x) and sigma_y > 0 and sigma_x > 0
    )
    return BeamShape(cy, cx, sigma_y, sigma_x, peak, success)


def aggregate_shapes(shapes: list[BeamShape]) -> BeamShape | None:
    """Aggregate successful beam shapes via the median of each parameter.

    Args:
        shapes: Per-frame beam shapes (failed ones are ignored).

    Returns:
        The aggregated shape, or None if no shape succeeded.
    """
    good = [s for s in shapes if s.success]
    if not good:
        return None
    return BeamShape(
        centroid_y=float(np.median([s.centroid_y for s in good])),
        centroid_x=float(np.median([s.centroid_x for s in good])),
        sigma_y=float(np.median([s.sigma_y for s in good])),
        sigma_x=float(np.median([s.sigma_x for s in good])),
        amplitude=float(np.median([s.amplitude for s in good])),
        success=True,
    )


def roi_from_shape(
    shape: BeamShape, n_sigma: float, *, minimum: int = 3
) -> tuple[int, int]:
    """Return a rectangular ``(height, width)`` spanning +/- ``n_sigma``.

    Args:
        shape: The aggregated beam shape.
        n_sigma: Half-extent of the ROI in beam sigmas.
        minimum: Smallest allowed ROI dimension in pixels.

    Returns:
        The ``(roi_height, roi_width)`` in pixels.
    """
    height = max(minimum, int(np.ceil(2 * n_sigma * shape.sigma_y)))
    width = max(minimum, int(np.ceil(2 * n_sigma * shape.sigma_x)))
    return height, width
