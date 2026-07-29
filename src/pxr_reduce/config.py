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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pxr_reduce.detectors import DetectorSpec, get_detector

DarkSide = Literal["LHS", "RHS"]
BeamLocator = Literal["peak", "centroid"]


def _serializable(data: dict[str, Any]) -> dict[str, Any]:
    """Convert values that JSON and TOML writers cannot represent.

    ``Path`` fields have to become strings before the config is embedded in an export
    header or written with :func:`json.dumps`, both of which reject them.
    """
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


@dataclass
class ReductionConfig:
    """Parameters controlling a PXR reduction run.

    Values are grouped by reduction stage. Defaults reproduce the historical
    behaviour of the original loader. The detector is referenced by name (or a
    :class:`DetectorSpec` instance) and resolved via :meth:`detector_spec`.

    Args:
        detector: Registered detector name or a ``DetectorSpec`` instance.
        header: Directory of beamline scan header files whose ``DATA`` section
            overrides the per-frame FITS metadata (see
            :mod:`pxr_reduce.header_file`). ``None`` — the default — leaves the FITS
            metadata untouched and the reduction unchanged.
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
        dezinger_threshold: A pixel is replaced by its local median when it
            exceeds this multiple of that median. Lower values remove hot pixels
            more aggressively (10 = replace pixels >10x the local median).
        mask_threshold: Counts marking likely beam locations for masking.
        mask_max_frames: Cap on how many frames are read to build the integration
            mask; frames are evenly subsampled above this count (0 = use all).
        drift_distance: Beam search radius (pixels): how far the beam may move
            between consecutive frames. Beam tracking searches within this radius
            of the previous frame's position.
        dark_pix_offset: Pixel offset of the dark ROI from the beam ROI.
        darkside: Preferred side ("LHS"/"RHS") to sample the dark ROI.
        saturate_threshold: Distance (ADU) from detector saturation that flags a
            frame as saturated.
        beam_snr_min: Minimum peak-to-noise ratio for a frame's beam to be
            considered detected; below this the frame is a dropout and its
            position is interpolated from the trajectory.
        track_smoothing: If True, fit a smooth trajectory per scan and replace
            dropouts/outliers with interpolated positions.
        track_poly_order: Polynomial order for the trajectory fit.
        centroid_radius: Half-size (px) of the window around the beam peak used to
            compute the centroid position. Smaller values lock more tightly to the
            peak so nearby scatter cannot pull the ROI off the beam.
        stitch_cutoff: Minimum spot/dark ratio for a point to be stitch-eligible.
        stitch_mark_tol: Minimum tracked-motor motion marking a stitch boundary.
        new_scan_marker: ``sam_th`` jump (deg) that marks the start of a new scan.
        drop_failed_stitch: Drop points that fail to stitch.
        stitch_condition_columns: Metadata columns whose change (beyond
            ``stitch_condition_tol``) marks a stitch boundary, alongside a
            ``sam_th`` back-step. Missing columns are ignored.
        stitch_condition_tol: A watched column must change by more than this to
            count as a condition change (metadata is pre-rounded, so 0.0 means
            "any real change").
        stitch_theta_backstep: A ``sam_th`` decrease larger than this (deg)
            between consecutive reflectivity frames marks a stitch boundary.
        stitch_normalized_conditions: Watched columns whose effect is already
            divided out of ``counts_refl``. A boundary triggered only by these (or
            by a bare back-step) must therefore fit a scale of ~1.0, which makes it
            checkable — see ``stitch_max_scale_deviation``. ``exposure`` qualifies
            because reflectivity is normalized by exposure x beam current.
        stitch_max_overlap_rms: Flag a stitch as suspect when its overlap points
            disagree about the fitted scale by more than this relative RMS
            (0.20 = 20%). Diagnostic only; nothing is dropped.
        stitch_max_scale_deviation: Flag a stitch as suspect when its fitted scale
            differs fractionally from the expected scale by more than this
            (0.10 = 10%), for boundaries where the expected scale is known.
        roi_from_beam_fit: If True, size the ROI from a moments fit of the direct
            beam (i0) frames instead of using ``roi_height``/``roi_width``.
        roi_n_sigma: ROI half-extent in beam sigmas when ``roi_from_beam_fit``.
        roi_fit_window: Window (px) around the beam peak used for the moments fit.
    """

    # --- Detector selection ---------------------------------------------------
    detector: str | DetectorSpec = "default"

    # --- Metadata source ------------------------------------------------------
    header: Path | None = None

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
    filter_size: int = 5
    dezinger: bool = True
    dezinger_threshold: float = 10.0
    mask_threshold: int = 80
    mask_max_frames: int = 200
    drift_distance: int = 45
    dark_pix_offset: int = 50
    darkside: DarkSide = "LHS"
    saturate_threshold: float = 2.0

    # --- Beam tracking --------------------------------------------------------
    beam_snr_min: float = 3.0
    track_smoothing: bool = True
    track_poly_order: int = 3
    centroid_radius: int = 8

    # --- ROI from direct-beam (i0) shape --------------------------------------
    roi_from_beam_fit: bool = False
    roi_n_sigma: float = 3.0
    roi_fit_window: int = 50

    # --- Stitching / scaling --------------------------------------------------
    stitch_cutoff: float = 1.003
    stitch_mark_tol: float = 1e-5
    new_scan_marker: float = 15.0
    drop_failed_stitch: bool = True

    # --- Stitch-boundary detection --------------------------------------------
    stitch_condition_columns: tuple[str, ...] = (
        "hos",
        "exposure",
        "slits_vert",
        "slits_horz",
    )
    stitch_condition_tol: float = 0.0
    stitch_theta_backstep: float = 0.001

    # --- Stitch-quality checks (diagnostic; never drop data) ------------------
    stitch_normalized_conditions: tuple[str, ...] = ("exposure",)
    stitch_max_overlap_rms: float = 0.20
    stitch_max_scale_deviation: float = 0.10

    def __post_init__(self) -> None:
        # Normalize to a tuple so a JSON round-trip (which yields a list) still
        # compares equal to the in-memory config.
        self.stitch_condition_columns = tuple(self.stitch_condition_columns)
        self.stitch_normalized_conditions = tuple(self.stitch_normalized_conditions)
        if self.header is not None:
            self.header = Path(self.header)
        if self.stitch_max_overlap_rms <= 0:
            raise ValueError("stitch_max_overlap_rms must be positive.")
        if self.stitch_max_scale_deviation <= 0:
            raise ValueError("stitch_max_scale_deviation must be positive.")
        if self.darkside not in ("LHS", "RHS"):
            raise ValueError(f"darkside must be 'LHS' or 'RHS', got {self.darkside!r}")
        if self.energy_resolution <= 0:
            raise ValueError("energy_resolution must be positive.")
        if self.roi_height <= 0 or self.roi_width <= 0:
            raise ValueError("roi_height and roi_width must be positive.")
        if self.centroid_radius <= 0:
            raise ValueError("centroid_radius must be positive.")
        if self.dezinger_threshold <= 1:
            raise ValueError(
                "dezinger_threshold must be > 1 (it is a multiple of the local "
                "median; values <= 1 would replace the beam itself)."
            )

    def detector_spec(self) -> DetectorSpec:
        """Resolve :attr:`detector` to a concrete :class:`DetectorSpec`."""
        return get_detector(self.detector)

    def to_header_dict(self) -> dict[str, Any]:
        """Return a flat, serializable mapping of config + detector for headers.

        The detector reference is expanded into its full specification so export
        headers capture every value used in the reduction.
        """
        data = _serializable(asdict(self))
        # Replace the detector reference with its expanded specification.
        data.pop("detector", None)
        data.update(self.detector_spec().to_header_dict())
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a round-trippable dict of all fields (detector as its name).

        Unlike :meth:`to_header_dict`, this preserves the ``detector`` field so
        the config can be reconstructed with :meth:`from_dict`.
        """
        data = _serializable(asdict(self))
        detector = self.detector
        data["detector"] = detector if isinstance(detector, str) else detector.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReductionConfig:
        """Construct a config from a dict produced by :meth:`to_dict`.

        Unknown keys are ignored so configs written by a newer version load
        (best-effort) under an older one.

        Args:
            data: Mapping of field names to values.

        Returns:
            The reconstructed configuration.
        """
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save_json(self, path: Path | str) -> Path:
        """Write the config to a JSON file (for the CLI ``--config`` option).

        Args:
            path: Destination path.

        Returns:
            The path written.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load_json(cls, path: Path | str) -> ReductionConfig:
        """Load a config from a JSON file written by :meth:`save_json`.

        Args:
            path: Path to the JSON file.

        Returns:
            The loaded configuration.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))
