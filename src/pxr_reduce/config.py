"""Typed reduction configuration.

:class:`ReductionConfig` replaces the untyped ``process_vars`` dictionary and the
property-generating decorator that backed the old loader. As a typed dataclass it
provides IDE autocompletion, validation, and — crucially — direct serialization
into export headers and CLI arguments.

Detector-specific constants (pixel size, bit depth, noise) do NOT live here; they
belong to :class:`~pxr_reduce.detectors.DetectorSpec`. The config only records
which detector to use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pxr_reduce.detectors import DetectorSpec, get_detector

DarkSide = Literal["LHS", "RHS"]


@dataclass
class ReductionConfig:
    """Parameters controlling a PXR reduction run.

    Values are grouped by reduction stage. Defaults reproduce the historical
    behaviour of the original loader. The detector is referenced by name (or a
    :class:`DetectorSpec` instance) and resolved via :meth:`detector_spec`.

    Args:
        detector: Registered detector name or a ``DetectorSpec`` instance.
        exposure_offset: Shutter open/close time added to exposure, in seconds.
        energy_resolution: Energy is rounded to ``round(E*res)/res`` eV.
        energy_offset: Additive energy correction in eV.
        sam_th_offset: Sample-theta offset in degrees; ``None`` triggers
            automatic determination when ``sam_th_correction`` is True.
        sam_th_correction: Auto-determine ``sam_th_offset`` from geometry.
        roi_height: ROI height in pixels for beam integration.
        roi_width: ROI width in pixels for beam integration.
        trim_x: Pixels trimmed from each vertical edge before processing.
        trim_y: Pixels trimmed from each horizontal edge before processing.
        filter_size: Median-filter kernel size for image cleanup.
        dezinger: If True, median-filter and dezinger each image; if False, skip
            it for a much faster (but noisier) reduction.
        mask_threshold: Counts marking likely beam locations for masking.
        mask_max_frames: Cap on how many frames are read to build the integration
            mask; frames are evenly subsampled above this count (0 = use all).
        drift_distance: Max beam drift radius (pixels) allowed between frames.
        dark_pix_offset: Pixel offset of the dark ROI from the beam ROI.
        darkside: Preferred side ("LHS"/"RHS") to sample the dark ROI.
        saturate_threshold: Distance (ADU) from detector saturation that flags a
            frame as saturated.
        stitch_cutoff: Minimum spot/dark ratio for a point to be stitch-eligible.
        stitch_mark_tol: Minimum tracked-motor motion marking a stitch boundary.
        new_scan_marker: ``sam_th`` jump (deg) that marks the start of a new scan.
        drop_failed_stitch: Drop points that fail to stitch.
        roi_from_beam_fit: If True, size the ROI from a moments fit of the direct
            beam (i0) frames instead of using ``roi_height``/``roi_width``.
        roi_n_sigma: ROI half-extent in beam sigmas when ``roi_from_beam_fit``.
        roi_fit_window: Window (px) around the beam peak used for the moments fit.
    """

    # --- Detector selection ---------------------------------------------------
    detector: str | DetectorSpec = "default"

    # --- Metadata / geometry --------------------------------------------------
    exposure_offset: float = 0.00389278
    energy_resolution: float = 20.0
    energy_offset: float = 0.0
    sam_th_offset: float | None = None
    sam_th_correction: bool = True

    # --- Image processing -----------------------------------------------------
    roi_height: int = 40
    roi_width: int = 40
    trim_x: int = 20
    trim_y: int = 20
    filter_size: int = 3
    dezinger: bool = True
    mask_threshold: int = 90
    mask_max_frames: int = 200
    drift_distance: int = 25
    dark_pix_offset: int = 20
    darkside: DarkSide = "LHS"
    saturate_threshold: float = 2.0

    # --- ROI from direct-beam (i0) shape --------------------------------------
    roi_from_beam_fit: bool = False
    roi_n_sigma: float = 3.0
    roi_fit_window: int = 50

    # --- Stitching / scaling --------------------------------------------------
    stitch_cutoff: float = 1.003
    stitch_mark_tol: float = 1e-5
    new_scan_marker: float = 15.0
    drop_failed_stitch: bool = True

    def __post_init__(self) -> None:
        if self.darkside not in ("LHS", "RHS"):
            raise ValueError(f"darkside must be 'LHS' or 'RHS', got {self.darkside!r}")
        if self.energy_resolution <= 0:
            raise ValueError("energy_resolution must be positive.")
        if self.roi_height <= 0 or self.roi_width <= 0:
            raise ValueError("roi_height and roi_width must be positive.")

    def detector_spec(self) -> DetectorSpec:
        """Resolve :attr:`detector` to a concrete :class:`DetectorSpec`."""
        return get_detector(self.detector)

    def to_header_dict(self) -> dict[str, Any]:
        """Return a flat, serializable mapping of config + detector for headers.

        The detector reference is expanded into its full specification so export
        headers capture every value used in the reduction.
        """
        data = asdict(self)
        # Replace the detector reference with its expanded specification.
        data.pop("detector", None)
        data.update(self.detector_spec().to_header_dict())
        return data
