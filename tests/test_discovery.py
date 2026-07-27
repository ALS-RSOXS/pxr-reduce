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
    "filename,expected",
    [
        ("B1A1_NEdge_XRR_89854-00001.fits", 89854),
        ("B1A1_XRR_P100_17344_000.fits", 17344),
        ("GlassA_90001-00000.fits", 90001),
        ("nodigits.fits", None),  # cannot tell scan from frame
    ],
)
def test_extract_scan_id_width_rule(filename, expected):
    assert extract_scan_id(filename) == expected


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
