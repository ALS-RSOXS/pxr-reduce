"""Image cleaning, masking, beam location, and ROI integration.

These functions replace the per-row ``apply(axis=1)`` image processing in the
original loader. They operate on plain arrays (fed from an
:class:`~pxr_reduce.io.fits_io.ImageStore`), so a reduction can stream one image
at a time and keep only scalar results.

Beam/ROI coordinates are expressed in the trimmed image frame produced by
:func:`clean_image`; keep that consistent across masking, beam-finding, and
integration.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, median_filter

from pxr_reduce.config import ReductionConfig
from pxr_reduce.detectors import DetectorSpec
from pxr_reduce.uncertainty import Value, net_counts
from pxr_reduce.utils.image import dezinger_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameIntegration:
    """Per-frame integration outputs.

    Args:
        beam_spot: ``(y, x)`` pixel of the beam centroid in the trimmed frame.
        counts_spot: Summed counts in the beam ROI (ADU).
        counts_dark: Summed counts in the dark ROI (ADU).
        net: Background-subtracted counts and uncertainty (ADU).
        counts_ratio: ``counts_spot / counts_dark`` (used for stitch eligibility).
        is_saturated: Whether the frame approached detector saturation.
    """

    beam_spot: tuple[int, int]
    counts_spot: float
    counts_dark: float
    net: Value
    counts_ratio: float
    is_saturated: bool


def trim(image: NDArray[np.floating], trim_x: int, trim_y: int) -> NDArray[np.floating]:
    """Remove ``trim_x``/``trim_y`` pixels from each image edge.

    Args:
        image: The image to trim.
        trim_x: Pixels to remove from each vertical edge (axis 0).
        trim_y: Pixels to remove from each horizontal edge (axis 1).

    Returns:
        The trimmed view of the image.
    """
    return image[trim_x:-trim_x, trim_y:-trim_y]


def dezinger(
    image: NDArray[np.floating], config: ReductionConfig
) -> NDArray[np.floating]:
    """Median-filter and dezinger an image in place of its own frame.

    Returns the input unchanged when ``config.dezinger`` is False. The image is
    expected to already be trimmed/cropped to the region of interest so the
    (expensive) median filter only runs where it is needed.

    Args:
        image: The image (or sub-region) to dezinger, in ADU.
        config: Reduction configuration (filter size, dezinger toggle).

    Returns:
        The dezingered image (same shape as input).
    """
    if not config.dezinger:
        return image
    filtered = median_filter(image, size=config.filter_size)
    return dezinger_image(image, med_result=filtered)


def clean_image(
    raw: NDArray[np.floating], config: ReductionConfig
) -> NDArray[np.floating]:
    """Trim and (optionally) dezinger a full raw image.

    Convenience for one-off full-frame cleaning (e.g. the viewer). The reduction
    hot path dezingers only the mask bounding box instead (see
    :func:`mask_bounding_box`).

    Args:
        raw: The raw image in ADU.
        config: Reduction configuration.

    Returns:
        The cleaned image in the trimmed frame.
    """
    return dezinger(trim(raw, config.trim_x, config.trim_y), config)


def build_series_mask(
    images: Iterable[NDArray[np.floating]], config: ReductionConfig
) -> NDArray[np.bool_]:
    """Build a static integration mask from the mean of a set of frames.

    Locates persistent hot regions (mean counts above ``mask_threshold``) and
    expands them by ``drift_distance`` (Euclidean) to allow for beam drift. The
    mean image is dezingered once so a stuck hot pixel does not seed the mask.

    The images are consumed as a stream; only a running sum is held in memory.

    Args:
        images: Iterable of frames (all the same shape), typically raw+trimmed.
        config: Reduction configuration (threshold and drift distance).

    Returns:
        Boolean mask, True where integration is permitted.

    Raises:
        ValueError: If no images are provided.
    """
    total: NDArray[np.floating] | None = None
    count = 0
    for image in images:
        if total is None:
            total = np.zeros_like(image, dtype=float)
        total += image
        count += 1
    if total is None or count == 0:
        raise ValueError("build_series_mask requires at least one image.")

    mean_image = dezinger(total / count, config)
    seeds = mean_image > config.mask_threshold
    if not seeds.any():
        return np.zeros_like(seeds, dtype=bool)
    # Euclidean dilation by drift_distance via a distance transform (O(N),
    # far faster than a large-footprint binary dilation).
    distance = distance_transform_edt(~seeds)
    return distance <= config.drift_distance


def mask_bounding_box(
    mask: NDArray[np.bool_], pad: int
) -> tuple[slice, slice]:
    """Return ``(row_slice, col_slice)`` bounding the True region plus padding.

    The beam-finder only searches inside the mask, so dezingering this box (which
    fully contains the mask) yields the same beam location as full-frame cleaning
    while filtering far fewer pixels. Padding leaves room for the beam/dark ROIs.

    Args:
        mask: The integration mask.
        pad: Pixels of margin added on every side (clipped to the image).

    Returns:
        Slices selecting the padded bounding box; the full frame if the mask is
        empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    r0 = max(0, int(r[0]) - pad)
    r1 = min(mask.shape[0], int(r[-1]) + pad + 1)
    c0 = max(0, int(c[0]) - pad)
    c1 = min(mask.shape[1], int(c[-1]) + pad + 1)
    return slice(r0, r1), slice(c0, c1)


def crop_window(
    image: NDArray[np.floating], center: tuple[int, int], size: int
) -> NDArray[np.floating]:
    """Return a ``size``x``size`` window centred on ``center`` (clipped to bounds).

    Args:
        image: Source image.
        center: ``(y, x)`` centre of the window.
        size: Desired side length in pixels.

    Returns:
        The cropped window (smaller than ``size`` near edges).
    """
    half = size // 2
    y, x = center
    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half + 1)
    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half + 1)
    return image[y0:y1, x0:x1]


def locate_beam(
    image: NDArray[np.floating], mask: NDArray[np.bool_]
) -> tuple[int, int]:
    """Locate the brightest pixel within the masked region.

    Args:
        image: The cleaned image.
        mask: Boolean mask constraining the search region.

    Returns:
        The ``(y, x)`` coordinate of the peak pixel.
    """
    masked = np.where(mask, image, 0)
    y, x = np.unravel_index(int(np.argmax(masked)), image.shape)
    return int(y), int(x)


def roi_slices(
    beam_spot: tuple[int, int], config: ReductionConfig
) -> tuple[slice, slice]:
    """Return ``(row_slice, col_slice)`` for the beam ROI around ``beam_spot``.

    Args:
        beam_spot: ``(y, x)`` beam centroid.
        config: Reduction configuration (ROI height/width).

    Returns:
        A ``(row_slice, col_slice)`` tuple for indexing the image.
    """
    h, w = config.roi_height, config.roi_width
    x_low = beam_spot[1] - w // 2
    y_low = beam_spot[0] - h // 2
    return slice(y_low, y_low + h), slice(x_low, x_low + w)


def dark_roi_slices(
    beam_spot: tuple[int, int], config: ReductionConfig
) -> tuple[slice, slice]:
    """Return ``(row_slice, col_slice)`` for the dark ROI beside the beam.

    Chooses the configured side when there is room, otherwise the opposite side.

    Args:
        beam_spot: ``(y, x)`` beam centroid.
        config: Reduction configuration (ROI size, dark offset, darkside).

    Returns:
        A ``(row_slice, col_slice)`` tuple for indexing the image.
    """
    h, w = config.roi_height, config.roi_width
    offset = config.dark_pix_offset
    if beam_spot[1] - 3 * w // 2 - offset > 0 and config.darkside == "LHS":
        x_low = beam_spot[1] - w // 2 - w - offset
    else:
        x_low = beam_spot[1] + w // 2 + offset
    y_low = beam_spot[0] - h // 2
    return slice(y_low, y_low + h), slice(x_low, x_low + w)


def integrate_frame(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    config: ReductionConfig,
    detector: DetectorSpec,
    exposure_s: float,
) -> FrameIntegration:
    """Locate the beam and integrate beam/dark ROIs for a single frame.

    Args:
        image: Cleaned image in the trimmed frame.
        mask: Series integration mask.
        config: Reduction configuration.
        detector: Detector supplying the noise model and saturation check.
        exposure_s: Exposure time in seconds (for the noise model).

    Returns:
        A :class:`FrameIntegration` with beam location, counts, and uncertainty.
    """
    beam_spot = locate_beam(image, mask)
    spot = image[roi_slices(beam_spot, config)]
    dark = image[dark_roi_slices(beam_spot, config)]
    counts_spot = float(spot.sum())
    counts_dark = float(dark.sum())
    net = net_counts(spot, dark, detector, exposure_s)
    ratio = counts_spot / counts_dark if counts_dark != 0 else np.inf
    return FrameIntegration(
        beam_spot=beam_spot,
        counts_spot=counts_spot,
        counts_dark=counts_dark,
        net=net,
        counts_ratio=ratio,
        is_saturated=detector.is_saturated(image, config.saturate_threshold),
    )
