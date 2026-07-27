"""Discover FITS scans under a parent folder and group them into samples.

Beamline FITS filenames carry two numbers: a static scan ID (a fixed-width block,
e.g. ``89854``) that identifies a whole theta sweep, and a frame index that
iterates as the last numeric block (e.g. ``00001``). Examples::

    B1A1_NEdge_XRR_89854-00001.fits   scan=89854  frame=1
    B1A1_XRR_P100_17344_000.fits      scan=17344  frame=0

- :func:`find_scan_files` collects every frame of a *known* scan ID (used when
  the ``[samples]`` map lists the scans to load). It matches the ID literally as
  a whole digit block, which cannot collide with the small frame index.
- :func:`discover_samples` and :func:`suggest_sample_map` scan a whole tree and
  report the distinct scan IDs present, to help build the config.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DIGITS = re.compile(r"\d+")


def extract_scan_id(
    filename: str, *, width: int = 5, regex: str | None = None
) -> int | None:
    """Extract the scan ID from a FITS filename.

    With ``regex``, the value of its ``scan`` capture group is used. Otherwise the
    last numeric block is treated as the (iterating) frame index and dropped, and
    the scan ID is the last remaining block of exactly ``width`` digits — or, if
    none has that width, the last remaining block.

    Args:
        filename: The filename (with or without directories/extension).
        width: Expected digit width of the scan-ID block.
        regex: Optional regex with a ``scan`` group overriding the width rule.

    Returns:
        The scan ID, or None if one cannot be determined.
    """
    stem = Path(filename).stem
    if regex is not None:
        match = re.search(regex, stem)
        if match is None:
            return None
        try:
            return int(match.group("scan"))
        except (IndexError, ValueError):
            return None

    blocks = _DIGITS.findall(stem)
    if len(blocks) < 2:
        # Need at least a scan block plus the frame index to tell them apart.
        return None
    candidates = blocks[:-1]  # drop the iterating frame index (last block)
    exact = [b for b in candidates if len(b) == width]
    chosen = exact[-1] if exact else candidates[-1]
    return int(chosen)


def _frame_index(filename: str) -> int:
    """Return the last numeric block of a filename (the iterating frame index)."""
    blocks = _DIGITS.findall(Path(filename).stem)
    return int(blocks[-1]) if blocks else 0


def find_scan_files(
    parent: Path | str,
    scan_id: int,
    *,
    glob: str = "*.fits",
    regex: str | None = None,
) -> list[Path]:
    """Return every FITS file belonging to ``scan_id`` under ``parent``.

    The tree is searched recursively, so a scan's frames may live in their own
    sub-folder or be mixed with others. Files are ordered by frame index.

    Args:
        parent: Parent directory to search recursively.
        scan_id: The scan ID to collect.
        glob: Glob for FITS files.
        regex: Optional scan-ID regex (see :func:`extract_scan_id`); when given,
            a file matches if its extracted scan ID equals ``scan_id``.

    Returns:
        The scan's FITS paths, ordered by frame index (possibly empty).
    """
    parent = Path(parent)
    if regex is not None:
        matches = [
            p
            for p in parent.rglob(glob)
            if extract_scan_id(p.name, regex=regex) == scan_id
        ]
    else:
        # Match the ID as a whole digit block; frame indices are small and cannot
        # equal a large scan ID, so this is unambiguous.
        pattern = re.compile(rf"(?<!\d){re.escape(str(scan_id))}(?!\d)")
        matches = [p for p in parent.rglob(glob) if pattern.search(p.name)]
    return sorted(matches, key=lambda p: _frame_index(p.name))


def discover_samples(
    parent: Path | str,
    *,
    glob: str = "*.fits",
    width: int = 5,
    regex: str | None = None,
) -> dict[int, list[Path]]:
    """Group every FITS file under ``parent`` by its scan ID.

    Args:
        parent: Parent directory to search recursively.
        glob: Glob for FITS files.
        width: Scan-ID digit width (see :func:`extract_scan_id`).
        regex: Optional scan-ID regex.

    Returns:
        Mapping of scan ID to its FITS paths (ordered by frame index), sorted by
        scan ID. Files whose scan ID cannot be extracted are skipped with a
        warning.
    """
    parent = Path(parent)
    groups: dict[int, list[Path]] = {}
    for path in parent.rglob(glob):
        scan_id = extract_scan_id(path.name, width=width, regex=regex)
        if scan_id is None:
            logger.warning("Could not extract a scan ID from %s; skipping.", path.name)
            continue
        groups.setdefault(scan_id, []).append(path)
    return {
        scan_id: sorted(files, key=lambda p: _frame_index(p.name))
        for scan_id, files in sorted(groups.items())
    }


def _sample_prefix(filename: str, scan_id: int) -> str:
    """Infer the sample-name prefix (the text before the scan ID)."""
    stem = Path(filename).stem
    match = re.search(rf"(?<!\d){re.escape(str(scan_id))}(?!\d)", stem)
    if match is None:
        return f"scan_{scan_id}"
    prefix = stem[: match.start()].rstrip(" _-")
    return prefix or f"scan_{scan_id}"


def suggest_sample_map(
    parent: Path | str,
    *,
    glob: str = "*.fits",
    width: int = 5,
    regex: str | None = None,
) -> dict[str, list[int]]:
    """Suggest a ``[samples]`` map by grouping discovered scans by name prefix.

    Scans whose filenames share a name prefix are grouped under that prefix, so
    repeats/energies of one sample cluster together for easy editing.

    Args:
        parent: Parent directory to search recursively.
        glob: Glob for FITS files.
        width: Scan-ID digit width.
        regex: Optional scan-ID regex.

    Returns:
        Mapping of inferred sample name to a sorted list of its scan IDs.
    """
    scans = discover_samples(parent, glob=glob, width=width, regex=regex)
    by_name: dict[str, list[int]] = {}
    for scan_id, files in scans.items():
        name = _sample_prefix(files[0].name, scan_id)
        by_name.setdefault(name, []).append(scan_id)
    return {name: sorted(ids) for name, ids in sorted(by_name.items())}
