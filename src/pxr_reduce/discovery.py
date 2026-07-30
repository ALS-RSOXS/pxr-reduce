"""Discover FITS scans under a parent folder and group them into samples.

Beamline filenames carry several numbers, and only two of them mean anything to a
reduction:

* a **scan ID** — the global identifier of one theta sweep, which is what
  ``[samples]`` lists, and which is *static* across every frame of that scan;
* a **frame index** — a sequential counter that *moves* from frame to frame.

Any other digits belong to the sample name (``TCTA_100_P100_``, ``T25``, ``B1A1``)
and track nothing. Naming conventions change between beamtimes, so nothing here
depends on the scan ID sitting in a particular position, on a digit width, or on a
separator. Instead:

* :func:`find_scan_files` **matches** the requested ID against every digit block in
  the filename, numerically — so a zero-padded ``002045`` in the filename matches
  ``2045`` in the config — and then identifies the frame counter as the one block
  that *moves* across the matched files, using it to order them.
* :func:`discover_samples` and :func:`suggest_sample_map` work the other way for
  config authoring: they group filenames by shape, find the moving block, and take
  the scan ID to be the block that is static within a scan but differs between
  scans.

Examples that all parse without configuration::

    TCTA_0_P100_ 002045 CCD 002.fits    scan=2045   frame=2
    From File Scan 006285 CCD 000.fits  scan=6285   frame=0
    T25 006288 CCD 001.fits             scan=6288   frame=1
    B1A1_XRR_P100_17344_000.fits        scan=17344  frame=0
    B1A1_NEdge_XRR_89854-00001.fits     scan=89854  frame=1

``scan_number_regex`` remains available for a convention these rules cannot handle.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_DIGITS = re.compile(r"\d+")


def digit_blocks(name: str) -> list[str]:
    """Return every digit block in a filename's stem, in order."""
    return _DIGITS.findall(Path(name).stem)


def _skeleton(name: str) -> str:
    """Return the filename with each digit block replaced by ``#``.

    Files sharing a skeleton were produced by the same naming convention, so their
    digit blocks are directly comparable position by position.
    """
    return _DIGITS.sub("#", Path(name).stem)


def _widest(blocks: list[list[str]], position: int) -> int:
    """Return the widest digit count seen at ``position`` (the tie-break metric).

    Scan IDs are wide, usually zero-padded blocks; numbers embedded in a sample name
    are short. When several blocks could be the scan ID, the widest is the better bet.
    """
    return max(len(b[position]) for b in blocks)


def _counter_score(values: set[int]) -> tuple[int, int, int]:
    """Rank how much a set of values looks like a sequential frame counter.

    A frame counter is dense (no gaps) and starts at 0 or 1. Scan IDs and
    sample-name numbers are neither, which separates them even when a small file set
    makes their value *counts* tie — two scans of two frames each would otherwise be
    indistinguishable by cardinality alone.
    """
    low, high = min(values), max(values)
    dense = len(values) == high - low + 1
    return (int(dense and low <= 1), int(dense), len(values))


def moving_block_position(
    names: Iterable[str], *, one_scan: bool = True
) -> int | None:
    """Return the position of the digit block that counts frames.

    Within a single scan the frame counter is the only block that changes. Across a
    whole naming-convention group the scan ID changes too, so the candidate that best
    resembles a dense zero-based counter wins (see :func:`_counter_score`).

    Args:
        names: Filenames to compare.
        one_scan: True when ``names`` is a single scan, where more than one varying
            block is unexpected and worth reporting.

    Returns:
        The block position of the frame counter, or None when nothing moves (a
        single-frame scan, or filenames with no digits).
    """
    blocks = [b for b in (digit_blocks(n) for n in names) if b]
    if not blocks:
        return None
    common = min(len(b) for b in blocks)
    varying = [p for p in range(common) if len({b[p] for b in blocks}) > 1]
    if not varying:
        return None

    scores = {p: _counter_score({int(b[p]) for b in blocks}) for p in varying}
    chosen = max(varying, key=lambda p: scores[p])
    if one_scan and len(varying) > 1:
        logger.warning(
            "%d digit blocks vary across these filenames (only the frame counter "
            "should); ordering by position %d.",
            len(varying),
            chosen,
        )
    elif not scores[chosen][0]:
        logger.debug(
            "No digit block looks like a dense zero-based frame counter; "
            "ordering by position %d.",
            chosen,
        )
    return chosen


def scan_block_position(names: Iterable[str], frame_position: int | None) -> int | None:
    """Return the position of the scan-ID block for one naming convention.

    The scan ID is static within a scan but differs between scans, so among the
    non-frame blocks it is the one whose value varies across the group. When several
    vary (a sample name that also counts up, e.g. ``T0``/``T5``/``T25``) or when the
    group holds a single scan and nothing varies, the widest block wins and the
    ambiguity is logged.

    Args:
        names: Filenames sharing one naming convention (see :func:`_skeleton`).
        frame_position: The frame counter's position, to exclude.

    Returns:
        The scan-ID block position, or None if the filenames have no usable blocks.
    """
    blocks = [digit_blocks(n) for n in names]
    blocks = [b for b in blocks if b]
    if not blocks:
        return None
    common = min(len(b) for b in blocks)
    positions = [p for p in range(common) if p != frame_position]
    if not positions:
        return None

    varying = [p for p in positions if len({b[p] for b in blocks}) > 1]
    candidates = varying or positions
    chosen = max(candidates, key=lambda p: _widest(blocks, p))
    if len(candidates) > 1:
        logger.debug(
            "Scan-ID block is ambiguous among positions %s; choosing the widest "
            "(position %d). Set scan_number_regex to be explicit.",
            candidates,
            chosen,
        )
    return chosen


def extract_scan_id(filename: str, *, regex: str | None = None) -> int | None:
    """Extract the scan ID from a single filename using an explicit regex.

    A lone filename does not say which of its numbers is the scan ID — that needs
    either a regex or a set of filenames to compare (see
    :func:`scan_block_position`). This helper therefore only handles the regex case.

    Args:
        filename: The filename (with or without directories/extension).
        regex: Regex exposing a ``scan`` capture group.

    Returns:
        The scan ID, or None if it cannot be determined.
    """
    if regex is None:
        return None
    match = re.search(regex, Path(filename).stem)
    if match is None:
        return None
    try:
        return int(match.group("scan"))
    except (IndexError, ValueError):
        return None


def find_scan_files(
    parent: Path | str,
    scan_id: int,
    *,
    glob: str = "*.fits",
    regex: str | None = None,
) -> list[Path]:
    """Return every FITS file belonging to ``scan_id`` under ``parent``.

    The tree is searched recursively, so a scan's frames may sit in their own
    sub-folder or be mixed with others. A file belongs to the scan when one of its
    digit blocks equals ``scan_id`` numerically, which makes zero-padding irrelevant:
    ``002045`` in a filename matches ``2045`` in the config. Files are then ordered by
    the block that moves across the matches — the frame counter.

    Args:
        parent: Parent directory to search recursively.
        scan_id: The scan ID to collect.
        glob: Glob for FITS files.
        regex: Optional regex with a ``scan`` group (see :func:`extract_scan_id`),
            for a convention the numeric match cannot handle.

    Returns:
        The scan's FITS paths, ordered by frame index (possibly empty).
    """
    paths = sorted(Path(parent).rglob(glob))
    if regex is not None:
        matches = [p for p in paths if extract_scan_id(p.name, regex=regex) == scan_id]
        return _ordered(matches)

    # Block positions are only comparable within one naming convention, so each
    # convention is analysed on its own -- over *all* its files, not just the matching
    # ones, so the frame counter is identifiable even for a single-frame match.
    found: list[tuple[int, str, Path]] = []
    shapes: list[str] = []
    for shape, members in _by_convention(paths).items():
        frame_position = moving_block_position(
            (p.name for p in members), one_scan=False
        )
        hits = [p for p in members if _has_static_id(p.name, scan_id, frame_position)]
        if not hits:
            continue
        shapes.append(shape)
        for path in hits:
            blocks = digit_blocks(path.name)
            index = 0 if frame_position is None else int(blocks[frame_position])
            found.append((index, path.name, path))

    if len(shapes) > 1:
        logger.warning(
            "Scan %s matches %d different filename patterns (%s); they will be "
            "pooled. Check that they are really the same scan.",
            scan_id,
            len(shapes),
            ", ".join(sorted(shapes)),
        )
    return [path for _, _, path in sorted(found, key=lambda item: item[:2])]


def _has_static_id(name: str, scan_id: int, frame_position: int | None) -> bool:
    """Whether ``name`` carries ``scan_id`` in a block that is *not* the frame counter.

    Matching numerically makes zero-padding irrelevant, but a long scan's frame index
    can coincide with another scan's ID — so a hit on the moving block is a
    coincidence, not membership.
    """
    return any(
        int(block) == scan_id
        for position, block in enumerate(digit_blocks(name))
        if position != frame_position
    )


def scan_ids_for(
    names: Iterable[str], *, regex: str | None = None
) -> dict[str, int | None]:
    """Map each filename to the scan ID it belongs to.

    Analyses the whole set: filenames are grouped by naming convention, the frame
    counter is found as the moving block, and the scan ID is the static block that
    differs between scans. A single filename cannot be resolved this way, which is why
    this takes a collection.

    Args:
        names: Filenames to classify. Several conventions may be mixed.
        regex: Optional scan-ID regex, used instead of the block analysis.

    Returns:
        Mapping of filename to scan ID, with None where it could not be determined.
    """
    resolved: dict[str, int | None] = {}
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(_skeleton(name), []).append(name)

    for shape, members in groups.items():
        if regex is not None:
            for name in members:
                resolved[name] = extract_scan_id(name, regex=regex)
            continue
        frame_position = moving_block_position(members, one_scan=False)
        scan_position = scan_block_position(members, frame_position)
        if scan_position is None:
            logger.warning(
                "Could not identify a scan-ID block for %d file(s) named like %r.",
                len(members),
                shape,
            )
            for name in members:
                resolved[name] = None
            continue
        for name in members:
            resolved[name] = int(digit_blocks(name)[scan_position])
    return resolved


def _by_convention(paths: list[Path]) -> dict[str, list[Path]]:
    """Group paths by naming convention so their digit blocks are comparable."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_skeleton(path.name), []).append(path)
    return groups


def discover_samples(
    parent: Path | str,
    *,
    glob: str = "*.fits",
    regex: str | None = None,
) -> dict[int, list[Path]]:
    """Group every FITS file under ``parent`` by its scan ID.

    Filenames are grouped by naming convention first, so that the moving (frame) and
    static (scan ID) blocks can be told apart by comparison rather than by assuming a
    position or width.

    Args:
        parent: Parent directory to search recursively.
        glob: Glob for FITS files.
        regex: Optional scan-ID regex, used instead of the block analysis.

    Returns:
        Mapping of scan ID to its FITS paths (ordered by frame index), sorted by scan
        ID. Files whose scan ID cannot be determined are skipped with a warning.
    """
    paths = sorted(Path(parent).rglob(glob))
    groups: dict[int, list[Path]] = {}

    for shape, members in _by_convention(paths).items():
        names = [p.name for p in members]
        if regex is None:
            # A convention group holds several scans, so the scan ID varies here too.
            frame_position = moving_block_position(names, one_scan=False)
            scan_position = scan_block_position(names, frame_position)
            if scan_position is None:
                logger.warning(
                    "Could not identify a scan-ID block for %d file(s) named like "
                    "%r; skipping. Set scan_number_regex to handle this convention.",
                    len(members),
                    shape,
                )
                continue
            ids = [int(digit_blocks(n)[scan_position]) for n in names]
        else:
            ids = [extract_scan_id(n, regex=regex) for n in names]

        for path, scan_id in zip(members, ids, strict=True):
            if scan_id is None:
                logger.warning(
                    "Could not extract a scan ID from %s; skipping.", path.name
                )
                continue
            groups.setdefault(scan_id, []).append(path)

    return {
        scan_id: _ordered(files) for scan_id, files in sorted(groups.items())
    }


def _ordered(files: list[Path]) -> list[Path]:
    """Order one scan's files by their frame counter."""
    position = moving_block_position(p.name for p in files)
    if position is None:
        return sorted(files)
    return sorted(files, key=lambda p: int(digit_blocks(p.name)[position]))


def _sample_prefix(filename: str, scan_id: int) -> str:
    """Infer the sample-name prefix: the text before the scan-ID block.

    The block is located by numeric value, so zero-padding in the filename does not
    prevent a match.
    """
    stem = Path(filename).stem
    for match in _DIGITS.finditer(stem):
        if int(match.group()) == scan_id:
            return stem[: match.start()].rstrip(" _-") or f"scan_{scan_id}"
    return f"scan_{scan_id}"


def suggest_sample_map(
    parent: Path | str,
    *,
    glob: str = "*.fits",
    regex: str | None = None,
) -> dict[str, list[int]]:
    """Suggest a ``[samples]`` map by grouping discovered scans by name prefix.

    Scans whose filenames share a name prefix are grouped under that prefix, so
    repeats/energies of one sample cluster together for easy editing.

    Args:
        parent: Parent directory to search recursively.
        glob: Glob for FITS files.
        regex: Optional scan-ID regex.

    Returns:
        Mapping of inferred sample name to a sorted list of its scan IDs.
    """
    scans = discover_samples(parent, glob=glob, regex=regex)
    by_name: dict[str, list[int]] = {}
    for scan_id, files in scans.items():
        by_name.setdefault(_sample_prefix(files[0].name, scan_id), []).append(scan_id)
    return {name: sorted(ids) for name, ids in sorted(by_name.items())}
