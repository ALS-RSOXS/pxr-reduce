import re
from typing import List, Optional


def infer_index_regex(
    filenames: List[str],
    *,
    index_group: str = "index",
    prefix_group: Optional[str] = None,
) -> str:
    """
    Infer a regular‑expression that captures the integer which increments by 1
    across a list of FITS filenames.

    Parameters
    ----------
    filenames : list[str]
        List of filenames (at least two entries). All must end with a similar extension –
        the suffix is not enforced, the algorithm works on any common pattern.

    index_group : str, optional
        Name of the capture group for the numeric index. Default: ``"index"``.

    prefix_group : str | None, optional
        If supplied, the literal part that appears **before** the index is also
        captured in a named group with this name.  If ``None`` (default) the
        prefix is left as a plain literal.

    Returns
    -------
    str
        A regex string suitable for ``re.search``. It is anchored (``^…$``) and
        contains the requested capture groups.

    Raises
    ------
    ValueError
        If the list is too short, contains no digit blocks, or no monotonic
        (+1) integer block can be identified.
    """
    if len(filenames) < 2:
        raise ValueError("At least two filenames are required to infer a sequence.")

    # ------------------------------------------------------------------
    # 0. Sort filenames using a natural/numeric sort so that, e.g.,
    #    "file_9.fits" comes before "file_10.fits" and "file_1000.fits"
    #    regardless of how the OS or glob returned them.
    # ------------------------------------------------------------------
    filenames = sorted(filenames, key=_natural_sort_key)

    # ------------------------------------------------------------------
    # 1. Locate every numeric substring in each filename
    # ------------------------------------------------------------------
    numeric_matches = [list(re.finditer(r"\d+", fn)) for fn in filenames]

    if any(not matches for matches in numeric_matches):
        raise ValueError("One or more filenames contain no digit blocks.")

    # ------------------------------------------------------------------
    # 2. Find the block that steps by +1 throughout the list
    # ------------------------------------------------------------------
    # Only compare blocks that exist in *all* filenames (use the minimum count)
    common_block_cnt = min(len(m) for m in numeric_matches)
    candidate_positions = []

    for pos in range(common_block_cnt):
        seq = [int(m[pos].group()) for m in numeric_matches]
        if all(seq[i + 1] - seq[i] == 1 for i in range(len(seq) - 1)):
            candidate_positions.append(pos)

    if not candidate_positions:
        raise ValueError("No integer block with a +1 progression found.")

    # Pick the candidate with the largest numeric span (most significant)
    best_pos = max(
        candidate_positions,
        key=lambda p: (
            int(numeric_matches[-1][p].group()) - int(numeric_matches[0][p].group())
        ),
    )

    # ------------------------------------------------------------------
    # 3. Build a regex from the first filename, replacing the chosen block
    # ------------------------------------------------------------------
    chosen_match = numeric_matches[0][best_pos]
    start, end = chosen_match.start(), chosen_match.end()

    # literal parts (escaped for regex)
    prefix_lit = re.escape(filenames[0][:start])
    suffix_lit = re.escape(filenames[0][end:])

    # ---- optional prefix capture -------------------------------------------------
    if prefix_group and prefix_lit:
        prefix_pat = f"(?P<{prefix_group}>{prefix_lit})"
    else:
        prefix_pat = prefix_lit

    # ---- index capture -----------------------------------------------------------
    widths = [len(m[best_pos].group()) for m in numeric_matches]
    if len(set(widths)) == 1:  # constant width → fixed quantifier
        width = widths[0]
        index_pat = rf"(?P<{index_group}>\d{{{width}}})"
    else:  # variable width → generic \d+
        index_pat = rf"(?P<{index_group}>\d+)"

    pattern = f"^{prefix_pat}{index_pat}{suffix_lit}$"
    return pattern
    
    
def _natural_sort_key(s: str):
    """
    Generate a sort key that orders strings with embedded integers numerically.
    e.g. ["file_9", "file_10", "file_100"] sorts correctly instead of
    lexicographically as ["file_10", "file_100", "file_9"].
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    ]
