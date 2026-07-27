"""Thin orchestrator tying the reduction pipeline together.

:class:`PXRLoader` replaces the monolithic ``PrsoxrLoader``. It holds:

* a small scalar metadata/counts table (:attr:`data`),
* an :class:`~pxr_reduce.io.fits_io.ImageStore` for lazy image access, and
* a :class:`~pxr_reduce.config.ReductionConfig`.

Heavy lifting is delegated to :mod:`pxr_reduce.metadata`,
:mod:`pxr_reduce.image_ops`, and :mod:`pxr_reduce.reduction`. Images never enter
the table; use :meth:`get_image` / :meth:`get_clean_image` to rebuild them for
debugging or visualization.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pxr_reduce import beam_fit, image_ops, metadata, reduction, tracking
from pxr_reduce.beam_fit import BeamShape
from pxr_reduce.config import ReductionConfig
from pxr_reduce.io.fits_io import ImageStore, read_fits_header
from pxr_reduce.utils import file_sort, name

logger = logging.getLogger(__name__)


def _subsample(indices: list[int], max_count: int) -> list[int]:
    """Return at most ``max_count`` evenly spaced indices (all if 0/None).

    Args:
        indices: Ordered frame indices.
        max_count: Maximum number to keep; 0 (or falsy) keeps all.

    Returns:
        The subsampled list, always including the endpoints of the input.
    """
    if not max_count or len(indices) <= max_count:
        return indices
    step = -(-len(indices) // max_count)  # ceil division
    return indices[::step]


class PXRLoader:
    """Load and reduce a series of PXR FITS images.

    Args:
        files: List of FITS file paths (str or Path).
        config: Reduction configuration; defaults to :class:`ReductionConfig`.
        auto_process: If True, run :meth:`process` immediately.
        cache_size: Max images held in the :class:`ImageStore` LRU cache.

    Raises:
        ValueError: If ``files`` is empty, contains a non-FITS file, or has no
            inferable numeric ordering.
        FileNotFoundError: If any path is not a file.
    """

    def __init__(
        self,
        files: list[str | Path],
        config: ReductionConfig | None = None,
        *,
        auto_process: bool = False,
        cache_size: int = 64,
    ) -> None:
        self.config = config or ReductionConfig()
        self.data_processed = False
        self.mask: NDArray[np.bool_] | None = None
        self.sam_th_offset_applied: float = 0.0
        self.beam_shape: BeamShape | None = None

        logger.info("Loading %d file(s) — reading headers...", len(files))
        paths = self._validate_files(files)
        index_regex = self._infer_naming(paths)
        self.name = self._infer_sample_name(paths[0], index_regex)
        self.path = paths[0].parent

        index_by_path = {
            p: int(re.search(index_regex, p.name).group("index")) for p in paths
        }
        self.files = sorted(paths, key=lambda p: index_by_path[p])
        self._store = ImageStore(
            {index_by_path[p]: p for p in self.files}, cache_size=cache_size
        )

        self.data = self._load_metadata(index_by_path)
        logger.info("Loaded %d frames for sample %r", len(self), self.name)

        if auto_process:
            self.process()

    # -- Construction helpers -------------------------------------------------

    @staticmethod
    def _validate_files(files: list[str | Path]) -> list[Path]:
        if not files:
            raise ValueError("The 'files' input is empty; nothing to load.")
        paths = file_sort.ensure_paths(files)
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"{p} is not a valid file.")
            if p.suffix != ".fits":
                raise ValueError(f"{p} is not a FITS file.")
        return paths

    @staticmethod
    def _infer_naming(paths: list[Path]) -> str:
        try:
            return name.infer_index_regex(
                [p.name for p in paths], prefix_group="re_sample_name"
            )
        except ValueError as exc:
            raise ValueError(
                "Filenames do not have an inferable numeric index for ordering. "
                f"Details: {exc}"
            ) from exc

    @staticmethod
    def _infer_sample_name(path: Path, index_regex: str) -> str:
        match = re.search(index_regex, path.name)
        return match.group("re_sample_name") if match else path.stem

    def _load_metadata(self, index_by_path: dict[Path, int]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for p in tqdm(self.files, desc="Loading FITS headers", unit="file"):
            record = read_fits_header(p)
            record["fits_index"] = index_by_path[p]
            records.append(record)
        table = metadata.build_metadata_table(records, self.config)
        table, self.sam_th_offset_applied = metadata.prepare_metadata(
            table, self.config
        )
        return table

    # -- Dunder / accessors ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.files)

    def __str__(self) -> str:
        return (
            f"PXRLoader(sample={self.name!r}, frames={len(self)}, "
            f"path={self.path}, processed={self.data_processed})"
        )

    def __call__(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for :meth:`reduce`."""
        return self.reduce(**kwargs)

    def path_for(self, fits_index: int) -> Path:
        """Return the FITS file path for a frame index."""
        return self._store.path(fits_index)

    def get_image(self, fits_index: int) -> NDArray[np.floating]:
        """Return the raw image for a frame index (lazy-loaded)."""
        return self._store.get(fits_index)

    def get_clean_image(self, fits_index: int) -> NDArray[np.floating]:
        """Return the trimmed/filtered/dezingered image for a frame index."""
        return image_ops.clean_image(self._store.get(fits_index), self.config)

    def query(self, **conditions: Any) -> pd.DataFrame:
        """Return metadata rows matching exact values or (low, high) ranges.

        Example:
            ``loader.query(energy=250.0, sam_th=(0.0, 5.0))``

        Args:
            **conditions: Column filters; a scalar matches equality, a 2-tuple
                matches an inclusive range.

        Returns:
            The matching subset of :attr:`data`.
        """
        mask = pd.Series(True, index=self.data.index)
        for column, condition in conditions.items():
            if column not in self.data.columns:
                raise KeyError(f"Unknown metadata column {column!r}.")
            if isinstance(condition, tuple) and len(condition) == 2:
                low, high = condition
                mask &= self.data[column].between(low, high)
            else:
                mask &= self.data[column] == condition
        return self.data[mask]

    # -- Processing -----------------------------------------------------------

    def process(self) -> None:
        """Track the beam per scan and integrate every frame.

        A coarse static mask is built from subsampled raw frames (for display and
        to bound the anchor search). Each scan is then tracked in acquisition
        order: an anchor is located on the bright first frame, and each subsequent
        frame's beam is found within ``drift_distance`` of the previous position
        via an SNR-gated centroid. Faint frames become dropouts whose positions
        are filled from a smoothed trajectory. Only a small region around the
        beam is dezingered per frame, so memory and time stay bounded.
        """
        detector = self.config.detector_spec()
        tx, ty = self.config.trim_x, self.config.trim_y
        all_indices = [int(i) for i in self.data["fits_index"]]
        exposures = dict(
            zip(self.data["fits_index"], self.data["exposure"], strict=True)
        )

        mask_indices = _subsample(all_indices, self.config.mask_max_frames)
        logger.info("Building integration mask from %d frame(s)", len(mask_indices))
        raw_stream = (
            image_ops.trim(self._store.get(i), tx, ty)
            for i in tqdm(mask_indices, desc="Building mask")
        )
        self.mask = image_ops.build_series_mask(raw_stream, self.config)

        # Bounding box of the swept-beam region; the anchor is located within the
        # mask so a weak first frame cannot lock onto a bright off-beam artifact.
        anchor_pad = self.config.dark_pix_offset + 2 * max(
            self.config.roi_height, self.config.roi_width
        )
        arow, acol = image_ops.mask_bounding_box(self.mask, anchor_pad)
        anchor_mask = self.mask[arow, acol]
        if not anchor_mask.any():
            anchor_mask = None

        if self.config.roi_from_beam_fit:
            self._maybe_fit_roi(arow, acol, detector)

        # Per-frame region big enough to contain the search radius + beam/dark ROI.
        region_half = (
            self.config.drift_distance
            + self.config.dark_pix_offset
            + 2 * max(self.config.roi_height, self.config.roi_width)
        )
        logger.info(
            "Tracking beam (search radius %d px) and integrating %d frame(s)...",
            self.config.drift_distance,
            len(all_indices),
        )

        # Direct-beam (i0) frames sit at a different detector position than the
        # specular track, so they are integrated at their own global peak; only
        # the reflectivity frames are tracked along the specular path.
        is_direct = dict(
            zip(
                (int(i) for i in self.data["fits_index"]),
                metadata.direct_beam_mask(self.data).to_numpy(),
            )
        )

        results: dict[int, image_ops.FrameIntegration] = {}
        track: dict[int, tuple[tuple[int, int], bool, float]] = {}
        for scan_id, group in self.data.groupby("scan", sort=True):
            scan_indices = [int(i) for i in group.sort_values("fits_index")["fits_index"]]
            direct_idx = [i for i in scan_indices if is_direct[i]]
            refl_idx = [i for i in scan_indices if not is_direct[i]]

            for idx in direct_idx:
                beam, snr, frame = self._integrate_direct(
                    idx, detector, float(exposures[idx])
                )
                results[idx], track[idx] = frame, (beam, True, snr)

            if refl_idx:
                prev = self._anchor(refl_idx[0], arow, acol, anchor_mask)
                for idx in tqdm(refl_idx, desc=f"Scan {scan_id}", leave=False):
                    beam, found, snr, frame = self._track_and_integrate(
                        idx, prev, region_half, detector, float(exposures[idx])
                    )
                    results[idx], track[idx] = frame, (beam, found, snr)
                    if found:
                        prev = beam
                if self.config.track_smoothing:
                    self._smooth_and_reintegrate(
                        refl_idx, track, results, region_half, detector, exposures
                    )

        self._assemble_counts([results[i] for i in all_indices])
        self.data["beam_found"] = [track[i][1] for i in all_indices]
        self.data["beam_snr"] = [track[i][2] for i in all_indices]
        self.data_processed = True

    def _integrate_direct(
        self, fits_index: int, detector: Any, exposure_s: float
    ) -> tuple[tuple[int, int], float, image_ops.FrameIntegration]:
        """Integrate a direct-beam (i0) frame at its own global peak.

        The direct beam (sample out) is intense and sits at a different detector
        position than the specular track, so it is located by the global peak of
        the whole frame rather than the tracked position. Returns
        ``(beam, snr, frame)`` with the beam in trimmed coordinates.
        """
        trimmed = image_ops.trim(
            self._store.get(fits_index), self.config.trim_x, self.config.trim_y
        )
        cleaned = image_ops.dezinger(trimmed, self.config)
        beam = tracking.anchor_position(cleaned)
        res = tracking.locate_in_window(
            cleaned,
            beam,
            self.config.drift_distance,
            snr_min=0.0,
            centroid_radius=self._centroid_radius(),
        )
        frame = image_ops.integrate_at(
            cleaned, beam, self.config, detector, exposure_s
        )
        frame = replace(frame, beam_spot=beam)
        return beam, res.snr, frame

    def _centroid_radius(self) -> int:
        """Half-size of the beam-centroid window (see ``config.centroid_radius``).

        Keeps the centroid local to the beam peak so off-beam scatter within the
        (larger) search radius does not pull the position.
        """
        return self.config.centroid_radius

    def _anchor(
        self,
        fits_index: int,
        arow: slice,
        acol: slice,
        anchor_mask: NDArray[np.bool_] | None,
    ) -> tuple[int, int]:
        """Locate the beam on the first frame of a scan to seed tracking.

        Searches within the swept-beam mask (bounding box) so a weak first frame
        cannot lock onto a brighter off-beam artifact. Returns the beam ``(y, x)``
        in trimmed coordinates.
        """
        trimmed = image_ops.trim(
            self._store.get(fits_index), self.config.trim_x, self.config.trim_y
        )
        cleaned = image_ops.dezinger(trimmed[arow, acol], self.config)
        y, x = tracking.anchor_position(cleaned, anchor_mask)
        return (y + arow.start, x + acol.start)

    def _track_and_integrate(
        self,
        fits_index: int,
        prev: tuple[int, int],
        region_half: int,
        detector: Any,
        exposure_s: float,
    ) -> tuple[tuple[int, int], bool, float, image_ops.FrameIntegration]:
        """Find the beam near ``prev`` and integrate; returns (beam, found, snr, frame)."""
        trimmed = image_ops.trim(self._store.get(fits_index), self.config.trim_x, self.config.trim_y)
        region, oy, ox = image_ops.crop_region(trimmed, prev, region_half)
        cleaned = image_ops.dezinger(region, self.config)
        center = (prev[0] - oy, prev[1] - ox)
        res = tracking.locate_in_window(
            cleaned,
            center,
            self.config.drift_distance,
            self.config.beam_snr_min,
            centroid_radius=self._centroid_radius(),
        )
        beam_region = (res.y, res.x) if res.found else (
            int(round(center[0])), int(round(center[1]))
        )
        frame = image_ops.integrate_at(
            cleaned, beam_region, self.config, detector, exposure_s
        )
        beam_trimmed = (beam_region[0] + oy, beam_region[1] + ox)
        frame = replace(frame, beam_spot=beam_trimmed)
        return beam_trimmed, res.found, res.snr, frame

    def _reintegrate_at(
        self,
        fits_index: int,
        beam_trimmed: tuple[int, int],
        region_half: int,
        detector: Any,
        exposure_s: float,
    ) -> image_ops.FrameIntegration:
        """Re-integrate a frame at a corrected (smoothed) beam position."""
        trimmed = image_ops.trim(self._store.get(fits_index), self.config.trim_x, self.config.trim_y)
        region, oy, ox = image_ops.crop_region(trimmed, beam_trimmed, region_half)
        cleaned = image_ops.dezinger(region, self.config)
        beam_region = (beam_trimmed[0] - oy, beam_trimmed[1] - ox)
        frame = image_ops.integrate_at(
            cleaned, beam_region, self.config, detector, exposure_s
        )
        return replace(frame, beam_spot=beam_trimmed)

    def _smooth_and_reintegrate(
        self,
        scan_indices: list[int],
        track: dict[int, tuple[tuple[int, int], bool, float]],
        results: dict[int, image_ops.FrameIntegration],
        region_half: int,
        detector: Any,
        exposures: dict[int, float],
    ) -> None:
        """Smooth a scan's trajectory and re-integrate corrected frames in place."""
        positions = [track[i][0] for i in scan_indices]
        found = [track[i][1] for i in scan_indices]
        corrected = tracking.smooth_track(
            positions, found, poly_order=self.config.track_poly_order
        )
        for idx, (newpos, changed) in zip(scan_indices, corrected):
            if changed:
                results[idx] = self._reintegrate_at(
                    idx, newpos, region_half, detector, float(exposures[idx])
                )
                track[idx] = (newpos, track[idx][1], track[idx][2])

    def _maybe_fit_roi(self, arow: slice, acol: slice, detector: Any) -> None:
        """Size the ROI from the direct-beam moments fit, if enabled."""
        shape = self._fit_beam_shape(arow, acol, self.mask[arow, acol], detector)
        if shape is not None:
            h, w = beam_fit.roi_from_shape(shape, self.config.roi_n_sigma)
            logger.info(
                "ROI from i0 beam fit: %d x %d px (sigma_y=%.2f, sigma_x=%.2f)",
                h, w, shape.sigma_y, shape.sigma_x,
            )
            self.config.roi_height, self.config.roi_width = h, w
            self.beam_shape = shape
        else:
            logger.warning(
                "Beam fit found no usable i0 frames; using configured ROI "
                "(%d x %d).",
                self.config.roi_height,
                self.config.roi_width,
            )

    def _fit_beam_shape(
        self,
        row_sl: slice,
        col_sl: slice,
        sub_mask: NDArray[np.bool_],
        detector: Any,
    ) -> BeamShape | None:
        """Fit the direct-beam (i0) frames and return the aggregated beam shape.

        Args:
            row_sl: Row slice of the mask bounding box.
            col_sl: Column slice of the mask bounding box.
            sub_mask: The mask cropped to the bounding box.
            detector: Detector spec (for the saturation check).

        Returns:
            The median beam shape over usable i0 frames, or None if none are
            usable (no direct-beam frames, all saturated, or all fits failed).
        """
        tx, ty = self.config.trim_x, self.config.trim_y
        i0 = metadata.direct_beam_mask(self.data)
        i0_indices = [int(v) for v in self.data.loc[i0, "fits_index"]]
        if not i0_indices:
            return None

        shapes: list[BeamShape] = []
        for i in i0_indices:
            sub = image_ops.trim(self._store.get(i), tx, ty)[row_sl, col_sl]
            cleaned = image_ops.dezinger(sub, self.config)
            if detector.is_saturated(cleaned, self.config.saturate_threshold):
                logger.debug("Skipping saturated i0 frame %d in beam fit", i)
                continue
            beam = image_ops.locate_beam(cleaned, sub_mask)
            window = image_ops.crop_window(cleaned, beam, self.config.roi_fit_window)
            shapes.append(beam_fit.estimate_moments(window))
        return beam_fit.aggregate_shapes(shapes)

    def _assemble_counts(self, results: list[image_ops.FrameIntegration]) -> None:
        """Fold per-frame integration results and normalization into the table."""
        exposure = self.data["exposure"].to_numpy()
        beam_current = self.data["beam_current"].to_numpy()
        norm = exposure * beam_current

        self.data["beam_spot"] = [r.beam_spot for r in results]
        self.data["counts_spot"] = [r.counts_spot for r in results]
        self.data["counts_dark"] = [r.counts_dark for r in results]
        self.data["counts_ratio"] = [r.counts_ratio for r in results]
        self.data["is_saturated"] = [r.is_saturated for r in results]
        net_value = np.array([r.net.value for r in results])
        net_sigma = np.array([r.net.sigma for r in results])
        self.data["counts_refl"] = net_value / norm
        self.data["counts_err"] = net_sigma / norm

    # -- Reduction ------------------------------------------------------------

    def reduce(
        self,
        *,
        apply_scale: bool = True,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Reduce the processed data to a 1D reflectivity dataset.

        Args:
            apply_scale: If False, skip stitch detection/scaling (quick preview).
            drop_duplicates: Average points sharing (sam_th, energy, polarization).

        Returns:
            The reduced dataset (columns: scan, energy, polarization, sam_th, q,
            R, R_err).

        Raises:
            RuntimeError: If :meth:`process` has not been run.
        """
        if not self.data_processed:
            raise RuntimeError("Call process() before reduce().")
        return reduction.reduce(
            self.data,
            self.config,
            apply_scale=apply_scale,
            drop_duplicates=drop_duplicates,
        )

    def diagnose_stitches(self) -> pd.DataFrame:
        """Return a per-boundary stitch diagnostic table for the processed data.

        Shows every detected stitch boundary, what triggered it, the changed
        conditions, the overlap-point count, and the fitted scale factor — for
        diagnosing a missing or mis-scaled stitch. See
        :func:`pxr_reduce.reduction.diagnose_stitches`.

        Returns:
            One row per stitch boundary (empty if none were detected).

        Raises:
            RuntimeError: If :meth:`process` has not been run.
        """
        if not self.data_processed:
            raise RuntimeError("Call process() before diagnose_stitches().")
        return reduction.diagnose_stitches(self.data, self.config)
