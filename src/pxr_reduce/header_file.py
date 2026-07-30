"""Override FITS metadata from the beamline's separate scan header files.

Some scans were collected with unreliable per-frame metadata in the FITS headers.
The beamline also writes a companion text file per scan holding the authoritative
motor record, and pointing ``reduction.header`` at the directory containing those
files replaces the affected columns.

File layout — three sections, of which only ``DATA`` is used:

.. code-block:: text

    HEADER          scan-level JSON (ignored)
    FILE            per-frame goal positions (ignored)
    DATA            per-frame record, one row per frame

``DATA`` starts with a tab-separated column-name row followed by one row per frame.
Motors appear as a ``"<name> Goal"``/``"<name> Actual"`` pair, and the **last field of
each row is the FITS file** that row describes — which is what frames are matched on,
so subsampled loads and per-scan files both work without any naming convention.

Only motors that appear as a Goal/Actual pair are overridden; every other column
(exposure, slit apertures, beam current, I0, polarization) keeps its FITS value.
``Goal`` becomes the canonical column, so every correction that keys off nominal
positions — scan segmentation, the sample-theta offset, stitch-boundary detection and
overlap matching — uses it, while ``<column>_actual`` carries the readback that ``q``
is computed from. The pre-override FITS value is kept as ``<column>_fits`` so the
metadata bug that motivated this can still be audited.

.. warning::
    The column-name row in observed files is malformed: it repeats some names, so it
    has more entries than the rows have fields. :func:`parse_header_file` collapses
    consecutive duplicate names and then *requires* the result to match the field
    count, because a silent misalignment would attach the wrong angle to every frame.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pxr_reduce import metadata
from pxr_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

# Line that introduces the per-frame table.
_DATA_MARKER = "DATA"

# Suffixes naming the two readings of one motor.
_GOAL = "Goal"
_ACTUAL = "Actual"

# How many missing frames to name before truncating the error message.
_MAX_REPORTED = 10


@dataclass(frozen=True)
class HeaderRow:
    """One frame's record from a scan header file.

    Args:
        source: The header file this row came from.
        values: Numeric ``DATA`` columns, keyed by column name. Non-numeric columns
            (e.g. ``Time of Day``) are omitted.
    """

    source: Path
    values: dict[str, float]


@dataclass(frozen=True)
class OverrideReport:
    """What a header-file override actually changed.

    Args:
        directory: The header directory used.
        n_frames: Frames whose metadata was overridden.
        columns: Canonical column names that were overridden.
        sources: Header files that contributed, by name.
        n_unused_rows: Header rows with no matching loaded frame (normal when only
            part of a scan is loaded, e.g. under ``--subsample``).
        n_dropped_frames: Loaded frames with no header row, which were dropped from
            the reduction.
        dropped_frames: Filenames of those dropped frames.
    """

    directory: Path
    n_frames: int
    columns: tuple[str, ...]
    sources: tuple[str, ...]
    n_unused_rows: int
    n_dropped_frames: int = 0
    dropped_frames: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return a one-line human-readable summary."""
        text = (
            f"{self.n_frames} frame(s) overridden from {len(self.sources)} header "
            f"file(s) in {self.directory}; columns: {', '.join(self.columns)}"
        )
        if self.n_dropped_frames:
            text += (
                f"; {self.n_dropped_frames} frame(s) dropped for having no header row"
            )
        return text


def _fits_key(field: str) -> str:
    """Return the lookup key for a FITS path recorded in a header file.

    The recorded path is Windows-style and relative to the header file, so only the
    basename is meaningful; it is lowercased because the source platform's filenames
    are case-insensitive.
    """
    return field.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def _dedupe_consecutive(names: list[str]) -> list[str]:
    """Collapse runs of identical column names into one.

    Works around the malformed name row described in the module docstring. Only
    *consecutive* repeats are collapsed, so two genuinely distinct columns that happen
    to share a name but sit apart are left alone (and will trip the field-count check
    rather than misalign silently).
    """
    deduped: list[str] = []
    for name in names:
        if not deduped or deduped[-1] != name:
            deduped.append(name)
    return deduped


def parse_header_file(path: Path | str) -> dict[str, HeaderRow]:
    """Parse the ``DATA`` section of one scan header file.

    Args:
        path: The header file to read.

    Returns:
        Mapping of lowercased FITS basename to that frame's :class:`HeaderRow`.

    Raises:
        ValueError: If the file has no ``DATA`` section, the column-name row cannot be
            aligned to the data rows, or a row has an unexpected field count.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    marker = next(
        (i for i, line in enumerate(lines) if line.strip() == _DATA_MARKER), None
    )
    if marker is None or marker + 1 >= len(lines):
        raise ValueError(f"{path} has no {_DATA_MARKER} section.")

    names = _dedupe_consecutive(lines[marker + 1].rstrip("\n").split("\t"))
    rows: dict[str, HeaderRow] = {}
    for offset, line in enumerate(lines[marker + 2 :], start=marker + 3):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        # One unnamed trailing field: the FITS path this row describes.
        if len(fields) != len(names) + 1:
            raise ValueError(
                f"{path} line {offset}: expected {len(names) + 1} fields "
                f"({len(names)} named columns plus the FITS path) but found "
                f"{len(fields)}. The column-name row cannot be aligned to the data, "
                "so the metadata would be attached to the wrong frames."
            )
        values: dict[str, float] = {}
        # The final field is the FITS path, which has no column name.
        for name, field in zip(names, fields[: len(names)], strict=True):
            try:
                values[name] = float(field)
            except ValueError:
                continue  # non-numeric column (e.g. "Time of Day")
        rows[_fits_key(fields[-1])] = HeaderRow(source=path, values=values)

    if not rows:
        raise ValueError(f"{path} has an empty {_DATA_MARKER} section.")
    logger.debug("Parsed %d header row(s) from %s", len(rows), path.name)
    return rows


@lru_cache(maxsize=8)
def index_header_directory(directory: Path) -> dict[str, HeaderRow]:
    """Index every header file in ``directory`` by the FITS files it describes.

    Cached per process: a batch run reduces many samples against the same directory
    and would otherwise re-parse every header file for each one. Treat the result as
    read-only.

    Args:
        directory: Directory holding the ``*.txt`` scan header files.

    Returns:
        Mapping of lowercased FITS basename to its :class:`HeaderRow`.

    Raises:
        NotADirectoryError: If ``directory`` is not a directory.
        ValueError: If no header files are found, or two files describe the same FITS
            frame (which would make the correct record ambiguous).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"reduction.header must be a directory of scan header files; "
            f"{directory} is not a directory."
        )

    index: dict[str, HeaderRow] = {}
    files = sorted(directory.glob("*.txt"))
    for path in files:
        try:
            rows = parse_header_file(path)
        except ValueError as exc:
            logger.warning("Skipping unparsable header file %s: %s", path.name, exc)
            continue
        for key, row in rows.items():
            if key in index:
                raise ValueError(
                    f"FITS frame {key!r} is described by two header files: "
                    f"{index[key].source.name} and {path.name}. Remove the stale one "
                    "so the correct record is unambiguous."
                )
            index[key] = row

    if not index:
        raise ValueError(
            f"No usable scan header files found in {directory} "
            f"({len(files)} .txt file(s) examined)."
        )
    logger.info(
        "Indexed %d frame(s) from %d header file(s) in %s",
        len(index),
        len(files),
        directory,
    )
    return index


def override_columns(available: Iterable[str]) -> dict[str, str]:
    """Return the motors to override, as ``{raw header name: canonical column}``.

    A motor qualifies only when the header file carries *both* its ``Goal`` and
    ``Actual`` columns, and only when the reduction has a canonical column for it
    (:data:`pxr_reduce.metadata.HEADER_NAMES`). Deriving the set this way means a
    header file that starts recording another motor as a Goal/Actual pair is picked up
    without a code change, while single-valued columns keep their FITS values.

    Args:
        available: Column names present in the header file.

    Returns:
        Mapping of raw motor name to canonical column name.
    """
    columns = set(available)
    return {
        raw: canonical
        for raw, canonical in metadata.HEADER_NAMES.items()
        if f"{raw} {_GOAL}" in columns and f"{raw} {_ACTUAL}" in columns
    }


def _round_canonical(
    values: pd.Series, canonical: str, config: ReductionConfig
) -> pd.Series:
    """Round an overridden canonical column exactly as the FITS path would.

    Matching the FITS-path rounding matters for more than tidiness: stitch overlap
    points are paired by *exact* equality on ``sam_th``, so the override has to land on
    the same grid. ``_actual`` columns are deliberately left unrounded by the caller —
    rounding energy to ``energy_resolution`` would erase the readback that ``q`` needs.
    """
    if canonical == "energy":
        return metadata.round_energy(values, config.energy_resolution)
    decimals = metadata.HEADER_RESOLUTIONS.get(canonical)
    return values if decimals is None else values.round(decimals)


def apply_override(
    table: pd.DataFrame,
    filenames: dict[int, str],
    config: ReductionConfig,
) -> tuple[pd.DataFrame, OverrideReport]:
    """Override the metadata table's motor columns from the header directory.

    Args:
        table: Standardized metadata table from
            :func:`~pxr_reduce.metadata.build_metadata_table`, before
            :func:`~pxr_reduce.metadata.prepare_metadata`.
        filenames: Mapping of ``fits_index`` to that frame's FITS filename.
        config: Reduction configuration; ``config.header`` names the directory.

    Returns:
        A ``(table, report)`` tuple. For each overridden motor the canonical column
        holds the ``Goal`` value, ``<column>_actual`` the readback, and
        ``<column>_fits`` the original FITS value. Frames with no header row are
        dropped from the table — their motor positions were never logged, so they
        cannot be placed on the q axis — and counted in the report.

    Raises:
        ValueError: If ``config.header`` is unset, no loaded frame can be matched at
            all, or the header files record none of the reduction's motors.
        NotADirectoryError: If ``config.header`` is not a directory.
    """
    if config.header is None:
        raise ValueError("apply_override requires config.header to be set.")

    directory = Path(config.header)
    index = index_header_directory(directory)

    keys = [_fits_key(filenames[int(i)]) for i in table["fits_index"]]
    matched = [key in index for key in keys]
    dropped = [
        filenames[int(i)]
        for i, ok in zip(table["fits_index"], matched, strict=True)
        if not ok
    ]
    if dropped:
        shown = ", ".join(dropped[:_MAX_REPORTED])
        more = (
            ""
            if len(dropped) <= _MAX_REPORTED
            else f" (+{len(dropped) - _MAX_REPORTED} more)"
        )
        logger.warning(
            "%d of %d loaded frame(s) have no row in any header file under %s and are "
            "dropped from the reduction: %s%s",
            len(dropped),
            len(keys),
            directory,
            shown,
            more,
        )
    if not any(matched):
        raise ValueError(
            f"None of the {len(keys)} loaded frame(s) have a row in any header file "
            f"under {directory}. Check that reduction.header points at the right "
            "directory for this data."
        )

    table = table.loc[pd.Series(matched, index=table.index)].reset_index(drop=True)
    keys = [key for key, ok in zip(keys, matched, strict=True) if ok]
    rows = [index[key] for key in keys]
    # Only override motors present for every matched frame.
    common = set.intersection(*(set(row.values) for row in rows))
    motors = override_columns(common)
    if not motors:
        raise ValueError(
            f"No Goal/Actual motor pairs in the header files under {directory} match "
            "the reduction's metadata columns; nothing would be overridden."
        )

    for raw, canonical in sorted(motors.items(), key=lambda kv: kv[1]):
        goal = pd.Series(
            [row.values[f"{raw} {_GOAL}"] for row in rows], index=table.index
        )
        actual = pd.Series(
            [row.values[f"{raw} {_ACTUAL}"] for row in rows], index=table.index
        )
        if canonical in table.columns:
            table[f"{canonical}_fits"] = table[canonical]
        table[canonical] = _round_canonical(goal, canonical, config)
        table[f"{canonical}_actual"] = actual

    report = OverrideReport(
        directory=directory,
        n_frames=len(table),
        columns=tuple(sorted(motors.values())),
        sources=tuple(sorted({row.source.name for row in rows})),
        n_unused_rows=len(index) - len(set(keys)),
        n_dropped_frames=len(dropped),
        dropped_frames=tuple(dropped),
    )
    logger.info("Header override: %s", report.describe())
    return table, report
