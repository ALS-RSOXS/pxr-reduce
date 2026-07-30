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

# Columns the loader supplies alongside the raw headers. ``scan_id`` is the ID from
# ``[samples]`` (or inferred from the filename), and is the unit that per-scan
# corrections — notably the sample-theta offset — are keyed on.
DERIVED_COLUMNS: tuple[str, ...] = ("scan_id",)

# Monitor headers that older beamline configurations do not record per frame, with
# the neutral value substituted when they are absent. Everything else in
# HEADER_NAMES is required, because the reduction cannot proceed without it.
#
# "Beam Current" feeds ``counts_refl = net / (exposure * beam_current)``. A *constant*
# current cancels exactly in R (which divides by the mean of the direct-beam frames),
# so substituting 1.0 only loses the correction for ring-current *decay* across a
# scan. "AI 3 Izero" is sanitized and displayed but never used by the reduction.
OPTIONAL_HEADERS: dict[str, float] = {
    "Beam Current": 1.0,
    "AI 3 Izero": 1.0,
}

# What the caller loses when an optional monitor header is substituted.
_OPTIONAL_CONSEQUENCE: dict[str, str] = {
    "Beam Current": (
        "Counts are no longer normalized by ring current; a constant current "
        "cancels in R, but decay across the scan is not corrected."
    ),
    "AI 3 Izero": "This column is not used by the reduction (display only).",
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

    Monitor headers listed in :data:`OPTIONAL_HEADERS` are substituted with a neutral
    value (and warned about) when the FITS files do not record them, which older
    beamline configurations do not.

    Returns:
        DataFrame with standardized, rounded columns, sorted by ``fits_index``.

    Raises:
        KeyError: If a required header column is missing from the records.
    """
    df = pd.DataFrame(records)
    missing = set(HEADER_NAMES) - set(df.columns)
    required_missing = missing - set(OPTIONAL_HEADERS)
    if required_missing:
        raise KeyError(
            f"Records are missing required header keys: {sorted(required_missing)}"
        )
    for key in sorted(missing):
        df[key] = OPTIONAL_HEADERS[key]
        logger.warning(
            "FITS headers do not record %r; substituting %g. %s",
            key,
            OPTIONAL_HEADERS[key],
            _OPTIONAL_CONSEQUENCE.get(key, ""),
        )

    # Columns the loader derives rather than reads from a header, carried through the
    # header-name selection below.
    derived = [c for c in DERIVED_COLUMNS if c in df.columns]
    df = (
        df[list(HEADER_NAMES) + derived]
        .rename(columns=HEADER_NAMES)
        .round(HEADER_RESOLUTIONS)
    )
    df["energy"] = round_energy(df["energy"], config.energy_resolution)
    df = df.sort_values("fits_index", ignore_index=True)
    df.insert(1, "scan", 0)
    return df


def round_energy(energy: pd.Series, resolution: float) -> pd.Series:
    """Round energy to the nearest ``1/resolution`` eV step.

    Args:
        energy: Photon energies in eV.
        resolution: Steps per eV; energy is rounded to ``round(E*res)/res``.

    Returns:
        The rounded energies.
    """
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
    """Split the table into sweeps and number them.

    A sweep boundary is a large ``sam_th`` jump **or** a change of ``scan_id``. The
    scan-ID rule is the important one: it guarantees that two scans pooled into one
    sample can never share an I0 or be stitched to each other, whatever their angles.
    Relying on the angle jump alone is a trap — two scans meeting below
    ``new_scan_marker`` would silently merge into one sweep and be normalized and
    stitched as though they were a single measurement.

    Args:
        df: Metadata table. ``scan_id`` is used when present.
        config: Reduction configuration (for the new-scan marker threshold).

    Returns:
        A new DataFrame with an integer ``scan`` column (the internal grouping key,
        running across the whole table) and ``sweep`` (a 0-based ordinal *within* each
        scan ID, stable regardless of what else was pooled, used for output and
        filenames).
    """
    df = df.copy()
    new_scan = df["sam_th"].diff().abs() > config.new_scan_marker
    if "scan_id" in df.columns:
        new_scan |= df["scan_id"].ne(df["scan_id"].shift())
    # The first frame starts sweep 0 rather than ending a previous one.
    new_scan.iloc[0] = False
    df["scan"] = new_scan.cumsum().astype(int)

    if "scan_id" in df.columns:
        df["sweep"] = (
            df.groupby("scan_id")["scan"].rank(method="dense").astype(int) - 1
        )
    else:
        df["sweep"] = df["scan"]
    return df


def sweep_tag(
    scan_id: Any, sweep: Any, energy: float, polarization: float
) -> str:
    """Return the identifier used to name per-sweep outputs.

    Every artifact describing one sweep — reduced rows, plots, diagnostic folders —
    is keyed the same way, so a questionable curve can be traced straight to its
    files.

    Args:
        scan_id: The scan ID the sweep belongs to.
        sweep: 0-based sweep ordinal within that scan.
        energy: Photon energy of the sweep (eV).
        polarization: Polarization of the sweep.

    Returns:
        A string of the form ``id2045_sweep0_E283.5_P100``.
    """
    # Identifiers arrive as floats whenever they were read out of a mixed-dtype row
    # (``frame.iloc[0]`` upcasts), which would render "id2045.0".
    return (
        f"id{_as_int(scan_id)}_sweep{_as_int(sweep)}"
        f"_E{energy:g}_P{polarization:g}"
    )


def _as_int(value: Any) -> Any:
    """Return ``value`` as an int when it is integral, else unchanged."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def by_sweep(df: pd.DataFrame) -> list[tuple[Any, pd.DataFrame]]:
    """Group rows by sweep, using the most specific identifier the table carries.

    ``(scan_id, sweep)`` is preferred and is stable across runs; the running ``scan``
    label is the fallback for tables built before sweeps were numbered.

    Args:
        df: Any table carrying sweep identifiers.

    Returns:
        A list of ``(key, group)`` pairs, ordered by key.
    """
    for keys in (["scan_id", "sweep"], ["scan"], ["energy", "polarization"]):
        if all(key in df.columns for key in keys):
            return list(df.groupby(keys, sort=True))
    return [((), df)]


def sweep_tag_for(group: pd.DataFrame) -> str:
    """Return :func:`sweep_tag` for one sweep's rows, warning if it is not uniform."""
    for column in ("energy", "polarization"):
        if column in group.columns and group[column].nunique() > 1:
            logger.warning(
                "Sweep spans %d distinct %s values; naming it after the first.",
                group[column].nunique(),
                column,
            )
    return sweep_tag(
        group["scan_id"].iloc[0] if "scan_id" in group.columns else -1,
        group["sweep"].iloc[0] if "sweep" in group.columns else 0,
        float(group["energy"].iloc[0]),
        float(group["polarization"].iloc[0]),
    )


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
    # Positions of the frames *after* each move. Indexing positionally, and dropping a
    # move on the final frame, keeps a late sam_z move from running past the end: only
    # the first move anchors the geometry, but indexing with the whole set would raise.
    after_move = np.where(sam_z_move.to_numpy())[0] + 1
    after_move = after_move[after_move < len(df)]
    if len(after_move) == 0:
        raise IndexError("No sam_z movement found to anchor the theta-2theta geometry.")

    first = int(after_move[0])
    begin_refl_angle = df["sam_th"].iloc[first]
    begin_ccd_angle = df["det_th"].iloc[first]
    return float(np.round(begin_ccd_angle / 2 - begin_refl_angle, 4))


def offset_group_column(df: pd.DataFrame) -> str:
    """Return the column that groups frames sharing one sample-theta offset.

    The offset is an encoder-zero calibration fixed by an alignment, so it belongs to
    a scan ID: several scans pooled into one sample were aligned separately and do not
    share it. Falls back to the sweep label when scan IDs are unavailable.
    """
    if "scan_id" in df.columns and bool((df["scan_id"] >= 0).all()):
        return "scan_id"
    return "scan"


def apply_energy_and_theta(
    df: pd.DataFrame, config: ReductionConfig
) -> tuple[pd.DataFrame, dict[Any, float]]:
    """Apply energy and sample-theta corrections and compute wavelength and q.

    Applies the configured energy offset, resolves the sample-theta offset
    (using ``config.sam_th_offset`` when given, otherwise auto-determining it if
    ``config.sam_th_correction`` is set), then derives ``wavelength`` and ``q``.

    The offset is determined and applied **per scan** (see
    :func:`offset_group_column`). Pooling several scan IDs into one sample and applying
    a single offset would shift every scan but the first: measured offsets of +0.042 and
    -0.014 deg for two scans of one sample are typical, and a 0.056 deg error puts their
    curves on visibly different q grids.

    When a header-file override has supplied ``sam_th_actual``/``energy_actual`` (see
    :mod:`pxr_reduce.header_file`), ``q`` is computed from those readbacks while the
    offset is still determined from — and every other correction still keys off — the
    nominal ``Goal`` columns. The offset is a geometric correction to the encoder zero,
    so it is applied to both readings.

    Args:
        df: Metadata table.
        config: Reduction configuration.

    Returns:
        A ``(dataframe, offsets)`` tuple, where ``offsets`` maps each group key (scan
        ID, or sweep label as a fallback) to the offset applied, for the export header.
    """
    df = df.copy()
    df["energy"] = df["energy"] + config.energy_offset
    if "energy_actual" in df.columns:
        df["energy_actual"] = df["energy_actual"] + config.energy_offset

    group_column = offset_group_column(df)
    keys = list(df[group_column].unique())
    offsets: dict[Any, float] = {}

    if config.sam_th_offset is not None:
        offsets = {key: float(config.sam_th_offset) for key in keys}
    elif config.sam_th_correction:
        for key, group in df.groupby(group_column, sort=True):
            try:
                offsets[key] = determine_sam_th_offset(group.reset_index(drop=True))
            except IndexError:
                logger.warning(
                    "Could not determine a sam_th offset for %s %s (no sam_z move "
                    "found); using 0.0",
                    group_column,
                    key,
                )
                offsets[key] = 0.0
        logger.info(
            "sam_th offset not given; assuming theta-2theta geometry -> per-%s "
            "offset(s): %s",
            group_column,
            ", ".join(f"{key}={value:+.4f}" for key, value in sorted(offsets.items())),
        )
    else:
        offsets = {key: 0.0 for key in keys}

    shift = df[group_column].map(offsets).to_numpy(dtype=float)
    df["sam_th"] = df["sam_th"] + shift
    if "sam_th_actual" in df.columns:
        df["sam_th_actual"] = df["sam_th_actual"] + shift

    # q reflects where the motors actually were, when that is known.
    theta_column = "sam_th_actual" if "sam_th_actual" in df.columns else "sam_th"
    energy_column = "energy_actual" if "energy_actual" in df.columns else "energy"
    if theta_column != "sam_th" or energy_column != "energy":
        logger.info(
            "Computing q from %s and %s (readback), with corrections from the "
            "nominal columns.",
            theta_column,
            energy_column,
        )

    df["wavelength"] = units.energy_to_wavelength(df[energy_column].to_numpy())
    df["q"] = units.theta_to_q(
        np.deg2rad(df[theta_column].to_numpy()), df["wavelength"].to_numpy()
    )
    return df, offsets


def prepare_metadata(
    df: pd.DataFrame, config: ReductionConfig
) -> tuple[pd.DataFrame, dict[Any, float]]:
    """Run the full metadata preparation pipeline.

    Applies monitor cleanup, scan labeling, and energy/theta/q derivation in the
    correct order.

    Args:
        df: Raw standardized metadata table from :func:`build_metadata_table`.
        config: Reduction configuration.

    Returns:
        A ``(prepared_dataframe, offsets)`` tuple; see
        :func:`apply_energy_and_theta` for the offset mapping.
    """
    df = clean_monitors(df, config)
    df = label_scans(df, config)
    df, offsets = apply_energy_and_theta(df, config)
    return df, offsets
