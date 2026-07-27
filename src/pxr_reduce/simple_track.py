"""A minimal, from-scratch beam tracker (median filter + local argmax).

This is a self-contained *plug-in* alternative to the tracker in
:mod:`pxr_reduce.tracking` / :meth:`pxr_reduce.core.PXRLoader.process`. It makes
no changes to that code; import :class:`SimplePXRLoader` in place of
:class:`~pxr_reduce.core.PXRLoader` to try it.

The strategy is deliberately simple — one primitive, applied two ways:

1. **Seed** — the beam is the global peak of the *median-filtered* frame. Median
   filtering suppresses lone hot pixels, so the peak is the centre of the
   highest-signal *area*, not a single zinger. This is robust for the bright
   direct beam.
2. **Step** — every following frame is located the same way, but restricted to a
   window of ``search_radius`` pixels around the previous position: the beam
   drifts only slightly between sequential measurements.

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

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from pxr_reduce import image_ops, metadata
from pxr_reduce.core import PXRLoader
from pxr_reduce.utils.image import dezinger_image

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


class SimplePXRLoader(PXRLoader):
    """A :class:`~pxr_reduce.core.PXRLoader` that uses the simple beam tracker.

    Only :meth:`process` is overridden; metadata handling, integration, reduction
    and export are all inherited unchanged, so this differs from the base loader
    solely in how the beam centre is found. Use it as a drop-in replacement:

    >>> loader = SimplePXRLoader(files, config)
    >>> loader.process(search_radius=15)
    >>> reduced = loader.reduce()
    """

    def process(
        self,
        *,
        search_radius: int | None = None,
        filter_size: int | None = None,
        progress: bool = True,
        verbose: bool = False,
    ) -> None:
        """Track the beam per scan and integrate every frame.

        For each scan, the first frame is seeded with a global median-filter peak
        (the direct beam), the first reflection frame after the ``sam_z`` move is
        re-seeded the same way (the beam jumps there), and every other frame is
        located within ``search_radius`` of the previous position. Integration is
        identical to the base loader (dezinger + ROI sum), so only the beam
        centre changes.

        Progress is reported live so a slow run (large frames, slow disk) is
        visible rather than looking hung: a per-frame bar shows the current scan,
        index, beam position and method, and a per-stage timing breakdown is
        logged at the end. If a single frame hangs, the bar stops on its index.

        Args:
            search_radius: Local search radius in pixels; defaults to
                ``config.drift_distance``.
            filter_size: Median-filter kernel size; defaults to
                ``config.filter_size``.
            progress: Show the live per-frame progress bar.
            verbose: Log per-stage timings (load/median/locate/clean/integrate)
                for every frame. Noisy for long scans; use to profile a subset.
        """
        radius = self.config.drift_distance if search_radius is None else search_radius
        ksize = self.config.filter_size if filter_size is None else filter_size
        detector = self.config.detector_spec()
        tx, ty = self.config.trim_x, self.config.trim_y
        all_indices = [int(i) for i in self.data["fits_index"]]
        exposures = dict(
            zip(self.data["fits_index"], self.data["exposure"], strict=True)
        )
        is_direct = dict(
            zip(
                (int(i) for i in self.data["fits_index"]),
                metadata.direct_beam_mask(self.data).to_numpy(),
                strict=True,
            )
        )

        n_frames = len(all_indices)
        n_scans = self.data["scan"].nunique()
        logger.info(
            "Simple tracker starting: %d frame(s) in %d scan(s), search radius "
            "%d px, median filter %d.",
            n_frames, n_scans, radius, ksize,
        )
        # Per-stage wall-clock, to reveal which step dominates a slow run. The
        # median filter is the expensive step and is now computed ONCE per frame
        # (shared by beam-finding and dezingering).
        stage_s = {
            "load": 0.0, "median": 0.0, "locate": 0.0, "clean": 0.0,
            "integrate": 0.0,
        }
        shape_logged = False
        t_start = time.perf_counter()

        results: dict[int, image_ops.FrameIntegration] = {}
        bar = tqdm(
            total=n_frames, desc="Simple track", unit="frame", disable=not progress
        )
        for scan_id, group in self.data.groupby("scan", sort=True):
            scan_indices = [
                int(i) for i in group.sort_values("fits_index")["fits_index"]
            ]
            prev_beam: tuple[int, int] | None = None
            prev_idx: int | None = None
            for pos, idx in enumerate(scan_indices):
                t0 = time.perf_counter()
                raw = image_ops.trim(self._store.get(idx), tx, ty)
                t1 = time.perf_counter()
                if not shape_logged:
                    logger.info(
                        "First frame loaded: trimmed shape %s, dtype %s. "
                        "Median-filtered once per frame (find + dezinger share it).",
                        raw.shape, raw.dtype,
                    )
                    shape_logged = True

                # One median filter per frame, reused for both beam-finding and
                # dezingering (dezinger_image takes the precomputed median).
                smoothed = median_filter(raw, size=ksize)
                t2 = time.perf_counter()

                # Re-seed at scan start and at the direct->reflection transition;
                # otherwise search locally around the previous position.
                transition = (
                    prev_idx is not None
                    and is_direct[prev_idx]
                    and not is_direct[idx]
                )
                reseed = prev_beam is None or pos == 0 or transition
                beam = peak(smoothed) if reseed else peak_near(
                    smoothed, prev_beam, radius
                )
                t3 = time.perf_counter()

                if self.config.dezinger:
                    cleaned = dezinger_image(
                        raw,
                        med_result=smoothed,
                        threshold=self.config.dezinger_threshold,
                    )
                else:
                    cleaned = raw
                t4 = time.perf_counter()
                results[idx] = image_ops.integrate_at(
                    cleaned, beam, self.config, detector, float(exposures[idx])
                )
                t5 = time.perf_counter()

                stage_s["load"] += t1 - t0
                stage_s["median"] += t2 - t1
                stage_s["locate"] += t3 - t2
                stage_s["clean"] += t4 - t3
                stage_s["integrate"] += t5 - t4
                method = "seed" if reseed else "track"
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
            "(load %.1fs, median %.1fs, locate %.1fs, clean %.1fs, "
            "integrate %.1fs).",
            n_frames, total_s, stage_s["load"], stage_s["median"],
            stage_s["locate"], stage_s["clean"], stage_s["integrate"],
        )

        self._assemble_counts([results[i] for i in all_indices])
        self.data["beam_found"] = [True] * len(all_indices)
        self.data["beam_snr"] = [float("nan")] * len(all_indices)
        self.data_processed = True
