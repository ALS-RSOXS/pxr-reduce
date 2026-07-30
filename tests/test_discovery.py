"""Tests for scan discovery and grouping."""

from __future__ import annotations

import pytest

from pxr_reduce.discovery import (
    discover_samples,
    extract_scan_id,
    find_scan_files,
    suggest_sample_map,
)


@pytest.mark.parametrize(
    "filename",
    [
        "B1A1_NEdge_XRR_89854-00001.fits",
        "TCTA_0_P100_ 002045 CCD 002.fits",
        "nodigits.fits",
    ],
)
def test_extract_scan_id_needs_a_regex(filename):
    # A lone filename cannot say which of its numbers is the scan ID -- that takes a
    # regex, or a set of filenames to compare.
    assert extract_scan_id(filename) is None


def test_extract_scan_id_regex_override():
    got = extract_scan_id("weird99999x_00001.fits", regex=r"weird(?P<scan>\d{5})")
    assert got == 99999


def _write(path, n=3, sample="GlassA", scan=90001):
    path.mkdir(parents=True, exist_ok=True)
    for frame in range(n):
        (path / f"{sample}_{scan}-{frame:05d}.fits").write_bytes(b"x")


def test_find_scan_files_orders_by_frame(tmp_path):
    _write(tmp_path / "sub", n=4, scan=90001)
    found = find_scan_files(tmp_path, 90001)
    assert [p.name for p in found] == [
        "GlassA_90001-00000.fits",
        "GlassA_90001-00001.fits",
        "GlassA_90001-00002.fits",
        "GlassA_90001-00003.fits",
    ]


def test_find_scan_files_ignores_other_scans(tmp_path):
    _write(tmp_path / "a", n=3, scan=90001)
    _write(tmp_path / "b", n=5, scan=90002)
    assert len(find_scan_files(tmp_path, 90001)) == 3
    assert len(find_scan_files(tmp_path, 90002)) == 5
    assert find_scan_files(tmp_path, 99999) == []


def test_discover_samples_groups_across_subfolders(tmp_path):
    _write(tmp_path / "a", n=3, sample="GlassA", scan=90001)
    _write(tmp_path / "b", n=5, sample="GlassA", scan=90002)
    _write(tmp_path / "b", n=4, sample="B1A1_XRR_P100", scan=17344)
    scans = discover_samples(tmp_path)
    assert {sid: len(fs) for sid, fs in scans.items()} == {17344: 4, 90001: 3, 90002: 5}


def test_suggest_sample_map_groups_by_prefix(tmp_path):
    _write(tmp_path / "a", n=2, sample="GlassA", scan=90001)
    _write(tmp_path / "b", n=2, sample="GlassA", scan=90002)
    _write(tmp_path / "c", n=2, sample="B1A1_XRR_P100", scan=17344)
    assert suggest_sample_map(tmp_path) == {
        "B1A1_XRR_P100": [17344],
        "GlassA": [90001, 90002],
    }


# --- Legacy naming conventions -------------------------------------------------
# Space separators, a literal " CCD " token, six-digit zero-padded scan IDs,
# variable-width frame indices, digits inside the sample name, and an inconsistent
# trailing underscore. None of these are configured for.

def _write_legacy(folder, sample, scan, n, *, start=0):
    folder.mkdir(parents=True, exist_ok=True)
    for frame in range(start, start + n):
        (folder / f"{sample} {scan:06d} CCD {frame:03d}.fits").write_bytes(b"x")


@pytest.mark.parametrize(
    "sample,scan",
    [
        ("From File Scan", 6285),   # spaces in the sample name, no digits
        ("T0", 6286),               # digit in the sample name
        ("T25", 6288),              # two digits in the sample name
        ("TCTA_0_P100_", 2045),     # several digits, trailing underscore
        ("TCTA_50_P190", 2250),     # several digits, no trailing underscore
    ],
)
def test_find_scan_files_matches_zero_padded_legacy_names(tmp_path, sample, scan):
    # The config lists 2045; the filename holds "002045". Padding must not matter.
    _write_legacy(tmp_path / f"{sample} {scan:06d} Images", sample, scan, 4)
    found = find_scan_files(tmp_path, scan)
    assert len(found) == 4
    assert [p.name for p in found] == [
        f"{sample} {scan:06d} CCD {i:03d}.fits" for i in range(4)
    ]


def test_find_scan_files_orders_variable_width_frame_index(tmp_path):
    # Indices run 998..1002, so a lexical sort would put 1000 before 998.
    folder = tmp_path / "s"
    folder.mkdir()
    for frame in (998, 999, 1000, 1001, 1002):
        (folder / f"TCTA_0_P100_ 002045 CCD {frame}.fits").write_bytes(b"x")
    found = find_scan_files(tmp_path, 2045)
    assert [p.name.rsplit(" ", 1)[-1] for p in found] == [
        "998.fits", "999.fits", "1000.fits", "1001.fits", "1002.fits"
    ]


def test_find_scan_files_ignores_a_coincidental_frame_counter(tmp_path):
    # A long scan whose frame index reaches another scan's ID must not absorb it.
    _write_legacy(tmp_path / "long", "Long", 9999, 3, start=2045)
    _write_legacy(tmp_path / "short", "TCTA_0_P100_", 2045, 4)
    found = find_scan_files(tmp_path, 2045)
    assert {p.parent.name for p in found} == {"short"}
    assert len(found) == 4


def test_discover_samples_handles_mixed_legacy_conventions(tmp_path):
    _write_legacy(tmp_path / "a", "From File Scan", 6285, 3)
    _write_legacy(tmp_path / "b", "T0", 6286, 2)
    _write_legacy(tmp_path / "c", "T25", 6288, 2)
    _write_legacy(tmp_path / "d", "TCTA_100_P100_", 2036, 2)
    _write_legacy(tmp_path / "e", "TCTA_100_P100_", 2039, 2)
    _write_legacy(tmp_path / "f", "TCTA_50_P190", 2250, 2)

    scans = discover_samples(tmp_path)
    assert {sid: len(fs) for sid, fs in scans.items()} == {
        2036: 2, 2039: 2, 2250: 2, 6285: 3, 6286: 2, 6288: 2
    }


def test_suggest_sample_map_recovers_legacy_sample_names(tmp_path):
    # The old code produced 'scan_2036' etc. because it could not locate the padded ID.
    _write_legacy(tmp_path / "a", "TCTA_100_P100_", 2036, 2)
    _write_legacy(tmp_path / "b", "TCTA_100_P100_", 2039, 2)
    _write_legacy(tmp_path / "c", "T25", 6288, 2)
    _write_legacy(tmp_path / "d", "From File Scan", 6285, 2)

    assert suggest_sample_map(tmp_path) == {
        "From File Scan": [6285],
        "T25": [6288],
        "TCTA_100_P100": [2036, 2039],
    }


def test_scan_regex_still_overrides(tmp_path):
    _write_legacy(tmp_path / "a", "TCTA_0_P100_", 2045, 3)
    found = find_scan_files(tmp_path, 2045, regex=r"(?P<scan>\d{6}) CCD")
    assert len(found) == 3
