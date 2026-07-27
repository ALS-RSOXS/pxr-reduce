"""The standard beam tracker (median filter + local argmax).

:func:`simple_process` is what :meth:`pxr_reduce.core.PXRLoader.process` runs. The
older SNR-gated tracker lives in :mod:`pxr_reduce.tracking` and is reached via the
deprecated :meth:`~pxr_reduce.core.PXRLoader.process_snr`.

The strategy is deliberately simple — one primitive, applied two ways:

1. **Seed** — the beam is the global peak of the *median-filtered* frame. Median
   filtering suppresses lone hot pixels, so the peak is the centre of the
   highest-signal *area*, not a single zinger. This is robust for the bright
   direct beam.
2. **Step** — every following frame is located the same way, but restricted to a
   window of ``search_radius`` pixels around the previous position: the beam
   drifts only slightly between sequential measurements. Only a region around
   that window is median-filtered (not the whole frame), which is far cheaper for
   large detectors and gives identical results, since the region fully contains
   the search window and the beam/dark ROIs.

Re-seeding (global peak again) happens at the start of every scan and at the
``sam_z`` move within a scan, where the beam jumps from the direct-beam position
to the first specular reflection. There is no SNR gate, no dropout handling, and
no trajectory smoothing: the tracker always returns a position.

To avoid median-filtering each frame twice, the loader computes the median filter
once and shares it between beam-finding (:func:`peak` / :func:`peak_near`, which
take an already-filtered image) and dezingering (which accepts the precomputed
median). The pure functions operate on plain arrays and are unit-testable without
any I/O.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from pxr_reduce import image_ops, metadata
from pxr_reduce.utils.image import dezinger_image

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)


def peak(smoothed: NDArray[np.floating]) -> tuple[int, int]:
    """Return the ``(y, x)`` global argmax of an (already median-filtered) image.

    Args:
        smoothed: A median-filtered image.

    Returns:
        The ``(y, x)`` pixel of the peak, in ``smoothed`` coordinates.
    """
    y, x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    return int(y), int(x)


def peak_near(
    smoothed: NDArray[np.floating], center: tuple[int, int], radius: int
) -> tuple[int, int]:
    """Return the argmax within ``radius`` px of ``center`` in a smoothed image.

    Args:
        smoothed: A median-filtered image.
        center: ``(y, x)`` centre of the search window (the previous position).
        radius: Search radius in pixels (how far the beam may have moved).

    Returns:
        The ``(y, x)`` pixel of the peak, in ``smoothed`` coordinates; ``center``
        (rounded) if the window falls entirely outside the image.
    """
    cy, cx = int(round(center[0])), int(round(center[1]))
    y0 = max(0, cy - radius)
    y1 = min(smoothed.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(smoothed.shape[1], cx + radius + 1)
    window = smoothed[y0:y1, x0:x1]
    if window.size == 0:
        return cy, cx
    ly, lx = np.unravel_index(int(np.argmax(window)), window.shape)
    return int(ly + y0), int(lx + x0)


def locate_peak(
    image: NDArray[np.floating], *, filter_size: int
) -> tuple[int, int]:
    """Locate the beam as the global peak of the median-filtered image.

    Args:
        image: The image to search (typically a trimmed frame).
        filter_size: Median-filter kernel size; larger values average over a
            wider area and suppress more single-pixel noise.

    Returns:
        The ``(y, x)`` pixel of the peak, in ``image`` coordinates.
    """
    return peak(median_filter(image, size=filter_size))


def locate_peak_near(
    image: NDArray[np.floating],
    center: tuple[int, int],
    radius: int,
    *,
    filter_size: int,
) -> tuple[int, int]:
    """Locate the beam within ``radius`` pixels of ``center``.

    Median-filters the image and returns the peak within a ``(2*radius+1)``-side
    window around ``center`` (clipped to the image bounds). Restricting the search
    keeps the position from jumping to brighter features elsewhere in the frame.

    Args:
        image: The image to search (typically a trimmed frame).
        center: ``(y, x)`` centre of the search window (the previous position).
        radius: Search radius in pixels (how far the beam may have moved).
        filter_size: Median-filter kernel size.

    Returns:
        The ``(y, x)`` pixel of the peak, in ``image`` coordinates; ``center``
        (rounded) if the window falls entirely outside the image.
    """
    return peak_near(median_filter(image, size=filter_size), center, radius)


def track(
    frames: Iterable[NDArray[np.floating]],
    *,
    radius: int,
    filter_size: int,
    reseed_indices: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Track the beam across a sequence of frames.

    The first frame is seeded with :func:`locate_peak`; every subsequent frame is
    located with :func:`locate_peak_near` around the previous position, except at
    ``reseed_indices`` where :func:`locate_peak` is used again (for a known jump,
    e.g. the direct-beam-to-reflection transition).

    Args:
        frames: Frames in acquisition order (all the same shape).
        radius: Search radius for the local step.
        filter_size: Median-filter kernel size.
        reseed_indices: Frame positions (0-based) at which to re-seed globally.

    Returns:
        The per-frame ``(y, x)`` positions, in frame coordinates.
    """
    reseed = reseed_indices or set()
    positions: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    for i, frame in enumerate(frames):
        if prev is None or i == 0 or i in reseed:
            beam = locate_peak(frame, filter_size=filter_size)
        else:
            beam = locate_peak_near(frame, prev, radius, filter_size=filter_size)
        positions.append(beam)
        prev = beam
    return positions


def simple_process(
    loader: PXRLoader,
    *,
    search_radius: int | None = None,
    filter_size: int | None = None,
    progress: bool = True,
    verbose: bool = False,
) -> None:
    """Track the beam per scan and integrate every frame (the standard tracker).

    This is the implementation behind :meth:`pxr_reduce.core.PXRLoader.process`.

    Direct-beam (i0) frames and the first reflection frame after the ``sam_z``
    move are median-filtered over the *full* frame and located by the global peak
    (exact centre / recover after the beam jumps). Every other frame is
    median-filtered over only a region around the previous beam and located within
    ``search_radius`` of it. Because that region contains the search window and
    the beam/dark ROIs, the cropped result is identical to full-frame processing
    but much faster on large detectors.

    Progress is reported live so a slow run (large frames, slow disk) is visible
    rather than looking hung: a per-frame bar shows the current scan, index, beam
    position and method, and a per-stage timing breakdown is logged at the end. If
    a single frame hangs, the bar stops on its index.

    Args:
        loader: The :class:`~pxr_reduce.core.PXRLoader` to populate in place.
        search_radius: Local search radius in pixels; defaults to
            ``config.drift_distance``.
        filter_size: Median-filter kernel size; defaults to ``config.filter_size``.
        progress: Show the live per-frame progress bar.
        verbose: Log per-stage timings (load/median/locate/clean/integrate) for
            every frame. Noisy for long scans; use to profile a subset.
    """
    config = loader.config
    radius = config.drift_distance if search_radius is None else search_radius
    ksize = config.filter_size if filter_size is None else filter_size
    detector = config.detector_spec()
    tx, ty = config.trim_x, config.trim_y
    # Region median-filtered for a tracked frame: big enough to hold the search
    # window plus the beam/dark ROIs, so the crop yields the same smoothed values
    # (hence beam and counts) as filtering the full frame.
    region_half = (
        radius + config.dark_pix_offset + 2 * max(config.roi_height, config.roi_width)
    )
    all_indices = [int(i) for i in loader.data["fits_index"]]
    exposures = dict(
        zip(loader.data["fits_index"], loader.data["exposure"], strict=True)
    )
    is_direct = dict(
        zip(
            (int(i) for i in loader.data["fits_index"]),
            metadata.direct_beam_mask(loader.data).to_numpy(),
            strict=True,
        )
    )

    n_frames = len(all_indices)
    n_scans = loader.data["scan"].nunique()
    logger.info(
        "Simple tracker starting: %d frame(s) in %d scan(s), search radius "
        "%d px, median filter %d.",
        n_frames, n_scans, radius, ksize,
    )
    # Per-stage wall-clock, to reveal which step dominates a slow run. The median
    # filter is computed ONCE per frame (shared by beam-finding and dezingering).
    stage_s = {
        "load": 0.0, "median": 0.0, "locate": 0.0, "clean": 0.0, "integrate": 0.0,
    }
    shape_logged = False
    t_start = time.perf_counter()

    results: dict[int, image_ops.FrameIntegration] = {}
    bar = tqdm(
        total=n_frames, desc="Simple track", unit="frame", disable=not progress
    )
    for scan_id, group in loader.data.groupby("scan", sort=True):
        scan_indices = [
            int(i) for i in group.sort_values("fits_index")["fits_index"]
        ]
        prev_beam: tuple[int, int] | None = None
        prev_idx: int | None = None
        for pos, idx in enumerate(scan_indices):
            t0 = time.perf_counter()
            trimmed = image_ops.trim(loader._store.get(idx), tx, ty)
            t1 = time.perf_counter()
            if not shape_logged:
                logger.info(
                    "First frame: trimmed shape %s, dtype %s. Direct-beam/seed "
                    "frames filter the full frame; tracked frames filter only a "
                    "%dx%d region.",
                    trimmed.shape, trimmed.dtype,
                    2 * region_half + 1, 2 * region_half + 1,
                )
                shape_logged = True

            # Direct-beam and segment-start frames filter the full frame for a
            # global peak (exact centre / recover after a jump); tracked
            # reflectivity frames filter only a region around the previous beam,
            # which is far cheaper for large detectors.
            transition = (
                prev_idx is not None
                and is_direct[prev_idx]
                and not is_direct[idx]
            )
            full_frame = prev_beam is None or is_direct.get(idx, False) or transition

            if full_frame:
                smoothed = median_filter(trimmed, size=ksize)
                t2 = time.perf_counter()
                beam = peak(smoothed)
                t3 = time.perf_counter()
                cleaned = (
                    dezinger_image(
                        trimmed,
                        med_result=smoothed,
                        threshold=config.dezinger_threshold,
                    )
                    if config.dezinger
                    else trimmed
                )
                t4 = time.perf_counter()
                frame = image_ops.integrate_at(
                    cleaned, beam, config, detector, float(exposures[idx])
                )
            else:
                region, oy, ox = image_ops.crop_region(
                    trimmed, prev_beam, region_half
                )
                smoothed = median_filter(region, size=ksize)
                t2 = time.perf_counter()
                center = (prev_beam[0] - oy, prev_beam[1] - ox)
                beam_region = peak_near(smoothed, center, radius)
                beam = (beam_region[0] + oy, beam_region[1] + ox)
                t3 = time.perf_counter()
                cleaned = (
                    dezinger_image(
                        region,
                        med_result=smoothed,
                        threshold=config.dezinger_threshold,
                    )
                    if config.dezinger
                    else region
                )
                t4 = time.perf_counter()
                frame = image_ops.integrate_at(
                    cleaned, beam_region, config, detector, float(exposures[idx])
                )
            # Report the beam in trimmed coordinates regardless of crop origin.
            results[idx] = replace(frame, beam_spot=beam)
            t5 = time.perf_counter()

            stage_s["load"] += t1 - t0
            stage_s["median"] += t2 - t1
            stage_s["locate"] += t3 - t2
            stage_s["clean"] += t4 - t3
            stage_s["integrate"] += t5 - t4
            method = "seed" if full_frame else "track"
            if verbose:
                logger.info(
                    "frame %d (scan %s, %s) beam=%s | load=%.3fs median=%.3fs "
                    "locate=%.3fs clean=%.3fs integrate=%.3fs",
                    idx, scan_id, method, beam,
                    t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                )
            bar.set_postfix(
                scan=int(scan_id), idx=idx,
                beam=f"{beam[0]},{beam[1]}", step=method, refresh=False,
            )
            bar.update(1)
            prev_beam, prev_idx = beam, idx
    bar.close()

    total_s = time.perf_counter() - t_start
    logger.info(
        "Simple tracker done: %d frame(s) in %.1fs "
        "(load %.1fs, median %.1fs, locate %.1fs, clean %.1fs, integrate %.1fs).",
        n_frames, total_s, stage_s["load"], stage_s["median"],
        stage_s["locate"], stage_s["clean"], stage_s["integrate"],
    )

    loader._assemble_counts([results[i] for i in all_indices])
    loader.data["beam_found"] = [True] * len(all_indices)
    loader.data["beam_snr"] = [float("nan")] * len(all_indices)
    loader.data_processed = True


def __getattr__(name: str) -> Any:
    """Provide the deprecated ``SimplePXRLoader`` name lazily (now ``PXRLoader``).

    Imported lazily to avoid an import cycle (``core`` imports this module).
    """
    if name == "SimplePXRLoader":
        from pxr_reduce.core import PXRLoader

        return PXRLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
