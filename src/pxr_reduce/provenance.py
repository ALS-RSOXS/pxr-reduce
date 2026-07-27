"""Provenance capture for reduced datasets.

Collects the "where did this come from and how was it made" metadata that gets
written into export headers: software version, git commit, collection/reduction
timestamps, and a per-source summary of the reduction inputs.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pxr_reduce.io.fits_io import read_fits_header

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)

# FITS header keys that may carry a collection timestamp, in priority order.
_COLLECTION_TIME_KEYS = ("DATE-OBS", "DATE", "TIME")

UNCERTAINTY_MODEL_DESCRIPTION = (
    "Per-pixel variance = Poisson shot noise + read noise + dark current "
    "(from detector model); propagated through ROI summation, dark subtraction, "
    "I0 normalization, and stitch scaling."
)


def software_version() -> str:
    """Return the installed pxr-reduce version, or 'unknown'."""
    try:
        return version("pxr-reduce")
    except PackageNotFoundError:
        return "unknown"


def git_commit(cwd: Path | None = None) -> str | None:
    """Return the short git commit hash for ``cwd``, or None if unavailable.

    Best-effort: any failure (not a repo, git missing, timeout) returns None.

    Args:
        cwd: Directory to query; defaults to the current working directory.

    Returns:
        The short commit hash, or None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Could not determine git commit: %s", exc)
        return None


def _collection_time_range(files: list[Path]) -> tuple[str | None, str | None]:
    """Return (earliest, latest) collection timestamps as ISO strings.

    Prefers FITS header date keys from the first and last files; falls back to
    file modification times.
    """
    if not files:
        return None, None

    def _stamp(path: Path) -> str:
        header = read_fits_header(path)
        for key in _COLLECTION_TIME_KEYS:
            if key in header and header[key]:
                return str(header[key])
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()

    try:
        return _stamp(files[0]), _stamp(files[-1])
    except (OSError, ValueError) as exc:
        logger.warning("Could not determine collection time: %s", exc)
        return None, None


@dataclass
class SourceProvenance:
    """Provenance for a single reduced source (one loader).

    Args:
        sample_name: Inferred sample name.
        source_path: Directory the FITS files were loaded from.
        n_frames: Number of frames loaded.
        n_scans: Number of distinct scans.
        energies: Sorted unique photon energies present (eV).
        polarizations: Sorted unique polarizations present.
        sam_th_offset: Sample-theta offset applied (deg).
        config: Flattened reduction config + detector specification.
        collection_time_start: Earliest collection timestamp (ISO), if known.
        collection_time_end: Latest collection timestamp (ISO), if known.
    """

    sample_name: str
    source_path: str
    n_frames: int
    n_scans: int
    energies: list[float]
    polarizations: list[float]
    sam_th_offset: float
    config: dict[str, Any]
    collection_time_start: str | None = None
    collection_time_end: str | None = None


def build_source_provenance(
    loader: PXRLoader, reduced: Any = None
) -> SourceProvenance:
    """Capture provenance from a processed loader.

    Args:
        loader: A :class:`~pxr_reduce.core.PXRLoader` (processed or not).
        reduced: The reduced output table. When given, the reported energies and
            polarizations are taken from it (the values actually present in the
            data), not from every loaded frame — so dropped-frame energies (i0,
            saturated, failed-stitch) do not appear in the header.

    Returns:
        A populated :class:`SourceProvenance`.
    """
    data = loader.data
    src = reduced if reduced is not None and len(reduced) else data
    start, end = _collection_time_range(loader.files)
    config = loader.config.to_header_dict()
    # Record the beam shape when the ROI was sized from the direct-beam fit.
    beam_shape = getattr(loader, "beam_shape", None)
    if beam_shape is not None:
        config["beam_fit_sigma_y"] = round(beam_shape.sigma_y, 3)
        config["beam_fit_sigma_x"] = round(beam_shape.sigma_x, 3)
    return SourceProvenance(
        sample_name=loader.name,
        source_path=str(loader.path),
        n_frames=len(loader),
        n_scans=int(data["scan"].nunique()) if "scan" in data else 0,
        energies=sorted({round(float(e), 6) for e in src["energy"].unique()}),
        polarizations=sorted({round(float(p), 6) for p in src["polarization"].unique()}),
        sam_th_offset=float(loader.sam_th_offset_applied),
        config=config,
        collection_time_start=start,
        collection_time_end=end,
    )


@dataclass
class ReductionProvenance:
    """Top-level provenance shared by a (possibly combined) dataset.

    Args:
        reduction_time: ISO timestamp when the reduction was written.
        software_version: pxr-reduce version string.
        git_commit: Short git hash, if available.
        uncertainty_model: Human-readable description of the error model.
        sources: Per-source provenance entries.
    """

    reduction_time: str
    software_version: str
    git_commit: str | None
    uncertainty_model: str
    sources: list[SourceProvenance] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        sources: list[SourceProvenance],
        *,
        reduction_time: datetime | None = None,
        cwd: Path | None = None,
    ) -> ReductionProvenance:
        """Build reduction-level provenance.

        Args:
            sources: Per-source provenance entries.
            reduction_time: Timestamp to record; defaults to now.
            cwd: Directory for the git-commit lookup.

        Returns:
            A populated :class:`ReductionProvenance`.
        """
        stamp = (reduction_time or datetime.now()).isoformat()
        return cls(
            reduction_time=stamp,
            software_version=software_version(),
            git_commit=git_commit(cwd),
            uncertainty_model=UNCERTAINTY_MODEL_DESCRIPTION,
            sources=list(sources),
        )
