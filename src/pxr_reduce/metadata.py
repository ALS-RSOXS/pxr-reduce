"""Metadata table construction and cleanup.

Builds the small, scalar-only metadata DataFrame from raw FITS headers and
applies the geometry/monitor corrections previously buried in the loader's
``cleanup_metadata``. Every operation here is vectorized (no ``apply(axis=1)``)
and side-effect free at the module level so it can be unit-tested with a small
synthetic DataFrame.

Images are deliberately absent from this table; they live in an
:class:`~pxr_reduce.io.fits_io.ImageStore`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pxr_reduce.config import ReductionConfig
from pxr_reduce.utils import units

logger = logging.getLogger(__name__)

# Mapping of raw FITS/AI header names to standardized column names.
HEADER_NAMES: dict[str, str] = {
    "fits_index": "fits_index",
    "Beamline Energy": "energy",
    "EPU Polarization": "polarization",
    "Sample Theta": "sam_th",
    "CCD Theta": "det_th",
    "Sample X": "sam_x",
    "Sample Y": "sam_y",
    "Sample Z": "sam_z",
    "EXPOSURE": "exposure",
    "Higher Order Suppressor": "hos",
    "Upstream JJ Vert Aperture": "slits_vert",
    "Upstream JJ Horz Aperture": "slits_horz",
    "Beam Current": "beam_current",
    "AI 3 Izero": "i0",
}

# Decimal places to round each standardized column to (energy handled separately).
HEADER_RESOLUTIONS: dict[str, int] = {
    "exposure": 4,
    "sam_th": 4,
    "det_th": 4,
    "sam_x": 4,
    "sam_y": 4,
    "hos": 1,
    "polarization": 0,
    "slits_vert": 4,
    "slits_horz": 4,
}

# Minimum valid beam current [mA]; below this the monitor is treated as unset.
_MIN_BEAM_CURRENT = 50.0

# sam_z motion that indicates the sample moved into the beam (ends the i0 region).
SAM_Z_BEAM_MOVE = 0.1


def direct_beam_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask flagging direct-beam (i0) frames, evaluated per scan.

    Within each scan the direct-beam frames are those preceding the first
    ``sam_z`` move that brings the sample into the beam.

    Args:
        df: Metadata table containing ``scan`` and ``sam_z`` columns.

    Returns:
        Boolean Series aligned to ``df.index``; True for direct-beam frames.
    """
    mask = pd.Series(False, index=df.index)
    for _, group in df.groupby("scan"):
        moved = group["sam_z"].diff().abs() > SAM_Z_BEAM_MOVE
        moved_positions = np.where(moved.to_numpy())[0]
        # Direct-beam frames are strictly before the sample moves in; the move
        # frame itself is the first reflectivity point.
        cutoff = int(moved_positions[0]) if len(moved_positions) else 0
        mask.loc[group.index[:cutoff]] = True
    return mask


def build_metadata_table(
    records: list[dict[str, Any]], config: ReductionConfig
) -> pd.DataFrame:
    """Assemble the standardized metadata table from raw header records.

    Args:
        records: One dict of raw FITS header values per frame; each must contain
            a ``fits_index`` key and the raw header names in :data:`HEADER_NAMES`.
        config: Reduction configuration (for energy resolution).

    Returns:
        DataFrame with standardized, rounded columns, sorted by ``fits_index``.

    Raises:
        KeyError: If a required header column is missing from the records.
    """
    df = pd.DataFrame(records)
    missing = set(HEADER_NAMES) - set(df.columns)
    if missing:
        raise KeyError(f"Records are missing required header keys: {sorted(missing)}")

    df = df[list(HEADER_NAMES)].rename(columns=HEADER_NAMES).round(HEADER_RESOLUTIONS)
    df["energy"] = _round_energy(df["energy"], config.energy_resolution)
    df = df.sort_values("fits_index", ignore_index=True)
    df.insert(1, "scan", 0)
    return df


def _round_energy(energy: pd.Series, resolution: float) -> pd.Series:
    """Round energy to the nearest ``1/resolution`` eV step."""
    return np.round(energy * resolution) / resolution


def clean_monitors(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Sanitize monitor columns and fold in the exposure offset (vectorized).

    Replaces non-physical monitor readings with safe defaults and adds the
    shutter open/close offset to the exposure time.

    Args:
        df: Metadata table.
        config: Reduction configuration (for the exposure offset).

    Returns:
        A new DataFrame with cleaned ``beam_current``, ``i0``, and ``exposure``.
    """
    df = df.copy()
    df["beam_current"] = np.where(
        df["beam_current"] > _MIN_BEAM_CURRENT, df["beam_current"], 1.0
    )
    df["i0"] = np.where(df["i0"] > 0, df["i0"], 1.0)
    df["exposure"] = np.where(
        df["exposure"] > 0, df["exposure"] + config.exposure_offset, 1.0
    )
    return df


def label_scans(df: pd.DataFrame, config: ReductionConfig) -> pd.DataFrame:
    """Assign a scan index to each frame based on large ``sam_th`` jumps.

    Args:
        df: Metadata table.
        config: Reduction configuration (for the new-scan marker threshold).

    Returns:
        A new DataFrame with an integer ``scan`` column.
    """
    df = df.copy()
    new_scan = df["sam_th"].diff().abs() > config.new_scan_marker
    df["scan"] = new_scan.cumsum().astype(int)
    return df


def determine_sam_th_offset(df: pd.DataFrame) -> float:
    """Determine the sample-theta offset from the first data-collection geometry.

    Assumes a theta-2theta geometry: the offset is ``det_th/2 - sam_th`` at the
    first frame after ``sam_z`` first moves (the beam-unblocking move).

    Args:
        df: Metadata table containing ``sam_z``, ``sam_th``, and ``det_th``.

    Returns:
        The sample-theta offset in degrees, rounded to 4 places.

    Raises:
        IndexError: If no ``sam_z`` movement is found to anchor the geometry.
    """
    sam_z_move = df["sam_z"].diff().abs() > 0.0
    sam_z_move.iloc[0] = False
    move_index = df["sam_z"][sam_z_move].index + 1
    begin_refl_angle = df["sam_th"].loc[move_index].iloc[0]
    begin_ccd_angle = df["det_th"].loc[move_index].iloc[0]
    return float(np.round(begin_ccd_angle / 2 - begin_refl_angle, 4))


def apply_energy_and_theta(
    df: pd.DataFrame, config: ReductionConfig
) -> tuple[pd.DataFrame, float]:
    """Apply energy and sample-theta corrections and compute wavelength and q.

    Applies the configured energy offset, resolves the sample-theta offset
    (using ``config.sam_th_offset`` when given, otherwise auto-determining it if
    ``config.sam_th_correction`` is set), then derives ``wavelength`` and ``q``.

    Args:
        df: Metadata table.
        config: Reduction configuration.

    Returns:
        A ``(dataframe, sam_th_offset)`` tuple; the offset applied is returned so
        it can be recorded in the export header.
    """
    df = df.copy()
    df["energy"] = df["energy"] + config.energy_offset

    sam_th_offset = 0.0
    if config.sam_th_offset is not None:
        sam_th_offset = config.sam_th_offset
    elif config.sam_th_correction:
        try:
            sam_th_offset = determine_sam_th_offset(df)
            logger.info(
                "sam_th offset not given; assuming theta-2theta geometry -> "
                "offset determined to be %.4f deg",
                sam_th_offset,
            )
        except IndexError:
            logger.warning(
                "Could not determine sam_th offset (no sam_z move found); using 0.0"
            )
            sam_th_offset = 0.0
    df["sam_th"] = df["sam_th"] + sam_th_offset

    df["wavelength"] = units.energy_to_wavelength(df["energy"].to_numpy())
    df["q"] = units.theta_to_q(
        np.deg2rad(df["sam_th"].to_numpy()), df["wavelength"].to_numpy()
    )
    return df, sam_th_offset


def prepare_metadata(
    df: pd.DataFrame, config: ReductionConfig
) -> tuple[pd.DataFrame, float]:
    """Run the full metadata preparation pipeline.

    Applies monitor cleanup, scan labeling, and energy/theta/q derivation in the
    correct order.

    Args:
        df: Raw standardized metadata table from :func:`build_metadata_table`.
        config: Reduction configuration.

    Returns:
        A ``(prepared_dataframe, sam_th_offset)`` tuple.
    """
    df = clean_monitors(df, config)
    df = label_scans(df, config)
    df, sam_th_offset = apply_energy_and_theta(df, config)
    return df, sam_th_offset
