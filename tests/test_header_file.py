"""Tests for the header-file metadata override."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxr_reduce import header_file, metadata
from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader

# Motors written as a Goal/Actual pair, in DATA column order.
_PAIRED = [
    "Sample X", "Sample Y", "Sample Z", "Sample Theta", "CCD Theta",
    "Beamline Energy", "Higher Order Suppressor",
]
# Single-valued columns, which must NOT be overridden.
_SINGLE = ["Axis Photonique Counts", "EPU Polarization", "Beam Current", "AI 3 Izero"]


def _data_names(*, malformed: bool = True) -> list[str]:
    """Build the DATA column-name row.

    Real files duplicate ``Axis Photonique Counts`` and pad three trailing
    ``Axis Photonique`` entries, so the row has more names than the rows have fields.
    """
    names = ["Time of Day", "Time (s)"]
    for motor in _PAIRED:
        names += [f"{motor} Goal", f"{motor} Actual"]
    names += _SINGLE
    if malformed:
        # Duplicate the way the beamline writes it.
        names.insert(names.index("Axis Photonique Counts") + 1, "Axis Photonique Counts")
        names += ["Axis Photonique", "Axis Photonique", "Axis Photonique"]
    else:
        names.append("Axis Photonique")
    return names


def _data_row(fits_name: str, goals: dict[str, float], deltas: dict[str, float]) -> str:
    """Build one DATA row; ``deltas`` offsets the Actual reading from the Goal."""
    fields = ["00:17:46", "8.152"]
    for motor in _PAIRED:
        goal = goals[motor]
        fields += [f"{goal:.8f}", f"{goal + deltas.get(motor, 0.0):.8f}"]
    fields += ["356187721.0", "100.0", "500.9699", "0.00469354"]
    fields.append("NaN")  # unnamed trailing numeric column
    fields.append(rf"..\Some Images\{fits_name}")
    return "\t".join(fields)


def write_header_file(path, frames, *, malformed: bool = True) -> None:
    """Write a scan header file with HEADER/FILE/DATA sections.

    Args:
        path: Destination ``.txt`` path.
        frames: ``(fits_name, goals, deltas)`` per frame.
        malformed: Reproduce the duplicated column names seen in real files.
    """
    lines = ["HEADER", '{', '    "Scan Type":"From File Scan"', '}', "FILE", "ignored"]
    lines += ["DATA", "\t".join(_data_names(malformed=malformed))]
    lines += [_data_row(n, g, d) for n, g, d in frames]
    # CRLF like the real files. newline="" disables the platform translation that
    # would otherwise turn each \r\n into \r\r\n and inject blank lines.
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8", newline="")


_GOALS = {
    "Sample X": 36.1157, "Sample Y": 26.8995, "Sample Z": -3.0,
    "Sample Theta": 1.2263, "CCD Theta": 2.0, "Beamline Energy": 380.0,
    "Higher Order Suppressor": 9.5,
}


@pytest.fixture
def header_dir(tmp_path):
    """Return a factory writing header files into a directory."""
    directory = tmp_path / "headers"
    directory.mkdir()

    def _make(name: str, frames) -> None:
        write_header_file(directory / name, frames)

    return directory, _make


def test_parse_extracts_rows_keyed_by_fits_filename(tmp_path):
    path = tmp_path / "scan.txt"
    write_header_file(
        path,
        [("S_000.fits", _GOALS, {}), ("S_001.fits", _GOALS, {"Sample Theta": 0.01})],
    )
    rows = header_file.parse_header_file(path)

    assert set(rows) == {"s_000.fits", "s_001.fits"}
    row = rows["s_001.fits"]
    assert row.source == path
    assert row.values["Sample Theta Goal"] == pytest.approx(1.2263)
    assert row.values["Sample Theta Actual"] == pytest.approx(1.2363)


def test_parse_aligns_columns_despite_duplicated_names(tmp_path):
    # The malformed name row has more entries than the rows have fields. Without
    # de-duplication, EPU Polarization would pick up the neighbouring column's value.
    path = tmp_path / "scan.txt"
    write_header_file(path, [("S_000.fits", _GOALS, {})], malformed=True)
    values = header_file.parse_header_file(path)["s_000.fits"].values

    assert values["EPU Polarization"] == pytest.approx(100.0)
    assert values["Beam Current"] == pytest.approx(500.9699)
    assert values["AI 3 Izero"] == pytest.approx(0.00469354)


def test_parse_rejects_unalignable_rows(tmp_path):
    path = tmp_path / "scan.txt"
    write_header_file(path, [("S_000.fits", _GOALS, {})])
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] += "\textra"  # one field too many
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be aligned"):
        header_file.parse_header_file(path)


def test_parse_requires_a_data_section(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text("HEADER\n{}\nFILE\nnothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no DATA section"):
        header_file.parse_header_file(path)


def test_override_columns_selects_only_goal_actual_pairs(tmp_path):
    path = tmp_path / "scan.txt"
    write_header_file(path, [("S_000.fits", _GOALS, {})])
    values = header_file.parse_header_file(path)["s_000.fits"].values

    motors = header_file.override_columns(values)
    assert set(motors.values()) == {
        "sam_x", "sam_y", "sam_z", "sam_th", "det_th", "energy", "hos"
    }
    # Single-valued columns are never overridden, so exposure and the slit apertures
    # (absent from DATA entirely) keep their FITS values.
    assert "polarization" not in motors.values()
    assert "beam_current" not in motors.values()
    assert "exposure" not in motors.values()


def test_index_directory_rejects_duplicate_frames(header_dir):
    directory, make = header_dir
    make("a.txt", [("S_000.fits", _GOALS, {})])
    make("b.txt", [("S_000.fits", _GOALS, {})])
    header_file.index_header_directory.cache_clear()
    with pytest.raises(ValueError, match="two header files"):
        header_file.index_header_directory(directory)


def test_index_directory_requires_a_directory(tmp_path):
    path = tmp_path / "scan.txt"
    write_header_file(path, [("S_000.fits", _GOALS, {})])
    header_file.index_header_directory.cache_clear()
    with pytest.raises(NotADirectoryError, match="must be a directory"):
        header_file.index_header_directory(path)


def test_index_directory_spans_multiple_files(header_dir):
    directory, make = header_dir
    make("scan_a.txt", [("A_000.fits", _GOALS, {}), ("A_001.fits", _GOALS, {})])
    make("scan_b.txt", [("B_000.fits", _GOALS, {})])
    header_file.index_header_directory.cache_clear()
    index = header_file.index_header_directory(directory)
    assert set(index) == {"a_000.fits", "a_001.fits", "b_000.fits"}
    assert index["b_000.fits"].source.name == "scan_b.txt"


def _fits_table(n: int) -> pd.DataFrame:
    """A metadata table as build_metadata_table would produce it, with wrong values."""
    return pd.DataFrame(
        {
            "fits_index": list(range(n)),
            "scan": [0] * n,
            "energy": [999.0] * n,  # deliberately wrong: the collection bug
            "polarization": [100.0] * n,
            "sam_th": [99.0] * n,
            "det_th": [99.0] * n,
            "sam_x": [0.0] * n,
            "sam_y": [0.0] * n,
            "sam_z": [0.0] * n,
            "exposure": [0.5, 2.5][:n] if n <= 2 else [0.5] * n,
            "hos": [0.0] * n,
            "slits_vert": [0.1] * n,
            "slits_horz": [0.1] * n,
            "beam_current": [500.0] * n,
            "i0": [1.0] * n,
        }
    )


def test_apply_override_replaces_goal_keeps_actual_and_fits(header_dir):
    directory, make = header_dir
    make(
        "scan.txt",
        [
            ("S_000.fits", _GOALS, {"Sample Theta": 0.002, "Beamline Energy": 0.0019}),
            ("S_001.fits", _GOALS, {"Sample Theta": -0.001}),
        ],
    )
    header_file.index_header_directory.cache_clear()
    table = _fits_table(2)
    config = ReductionConfig(header=directory)

    out, report = header_file.apply_override(
        table, {0: "S_000.fits", 1: "S_001.fits"}, config
    )

    # Canonical columns now hold the Goal values (rounded as the FITS path rounds).
    assert out["sam_th"].tolist() == [1.2263, 1.2263]
    assert out["energy"].tolist() == [380.0, 380.0]
    # Actual is kept at full precision -- rounding energy would erase the readback.
    assert out["energy_actual"].iloc[0] == pytest.approx(380.0019)
    assert out["sam_th_actual"].iloc[0] == pytest.approx(1.2283)
    # The original FITS values are preserved so the collection bug stays auditable.
    assert out["sam_th_fits"].tolist() == [99.0, 99.0]
    assert out["energy_fits"].tolist() == [999.0, 999.0]
    # Columns without a Goal/Actual pair are untouched.
    assert out["exposure"].tolist() == [0.5, 2.5]
    assert out["slits_vert"].tolist() == [0.1, 0.1]
    assert out["polarization"].tolist() == [100.0, 100.0]

    assert report.n_frames == 2
    assert "sam_th" in report.columns and "energy" in report.columns
    assert report.sources == ("scan.txt",)


def test_apply_override_refuses_a_partial_match(header_dir):
    directory, make = header_dir
    make("scan.txt", [("S_000.fits", _GOALS, {})])
    header_file.index_header_directory.cache_clear()
    config = ReductionConfig(header=directory)

    with pytest.raises(ValueError, match="no row in any header file"):
        header_file.apply_override(
            _fits_table(2), {0: "S_000.fits", 1: "S_001.fits"}, config
        )


def test_apply_override_counts_unused_header_rows(header_dir):
    # Loading a subset of a scan is legitimate; the extra rows are just reported.
    directory, make = header_dir
    make("scan.txt", [(f"S_{i:03d}.fits", _GOALS, {}) for i in range(4)])
    header_file.index_header_directory.cache_clear()
    config = ReductionConfig(header=directory)

    _, report = header_file.apply_override(_fits_table(2), {0: "S_000.fits", 1: "S_001.fits"}, config)
    assert report.n_unused_rows == 2


def test_q_uses_actual_while_corrections_use_goal(header_dir):
    directory, make = header_dir
    # A 0.5 deg readback error would move q noticeably if q used Goal.
    make(
        "scan.txt",
        [("S_000.fits", _GOALS, {"Sample Theta": 0.5}),
         ("S_001.fits", _GOALS, {"Sample Theta": 0.5})],
    )
    header_file.index_header_directory.cache_clear()
    config = ReductionConfig(header=directory, sam_th_correction=False)
    table, _ = header_file.apply_override(
        _fits_table(2), {0: "S_000.fits", 1: "S_001.fits"}, config
    )
    out, _ = metadata.apply_energy_and_theta(table, config)

    # Goal drives the nominal angle; q is computed from the readback.
    assert out["sam_th"].iloc[0] == pytest.approx(1.2263)
    assert out["sam_th_actual"].iloc[0] == pytest.approx(1.7263)
    expected_q = 4 * np.pi * np.sin(np.deg2rad(1.7263)) / out["wavelength"].iloc[0]
    assert out["q"].iloc[0] == pytest.approx(expected_q)


def test_no_header_config_leaves_everything_unchanged(processed_loader_factory):
    loader = processed_loader_factory()
    assert loader.config.header is None
    assert loader.header_override is None
    # No override columns are added when the flag is absent.
    assert not [c for c in loader.data.columns if c.endswith(("_actual", "_fits"))]


def test_config_serializes_header_path_for_export(tmp_path):
    config = ReductionConfig(header=tmp_path / "headers")
    assert isinstance(config.header, type(tmp_path))
    # Both dict forms must be JSON/TOML-writable, so the Path becomes a string.
    assert config.to_dict()["header"] == str(tmp_path / "headers")
    assert config.to_header_dict()["header"] == str(tmp_path / "headers")
    assert ReductionConfig.from_dict(config.to_dict()).header == config.header


def test_loader_applies_override_end_to_end(
    tmp_path, frame_builders, fits_writer, header_dir
):
    beam_image, frame_header = frame_builders
    directory, make = header_dir

    # FITS headers carry a wrong sam_th; the header file has the true geometry.
    files = []
    frames = []
    for i, th in enumerate([0.0, 0.0, 1.2263, 1.2789, 1.3316, 1.3842]):
        header = frame_header(99.0, 1.0 if i >= 2 else 0.0)  # wrong angle in FITS
        files.append(
            fits_writer(tmp_path / f"MF999A_{i}.fits", beam_image(5000.0), header)
        )
        goals = dict(_GOALS, **{"Sample Theta": th, "CCD Theta": 2 * th,
                                "Sample Z": 1.0 if i >= 2 else 0.0})
        frames.append((f"MF999A_{i}.fits", goals, {"Sample Theta": 0.001}))
    make("MF999A.txt", frames)
    header_file.index_header_directory.cache_clear()

    loader = PXRLoader(
        files,
        ReductionConfig(roi_height=9, roi_width=9, header=directory,
                        sam_th_correction=False),
    )
    assert loader.header_override is not None
    assert loader.header_override.n_frames == 6
    # The wrong FITS angle is gone from the canonical column but still auditable.
    assert loader.data["sam_th"].tolist() == [0.0, 0.0, 1.2263, 1.2789, 1.3316, 1.3842]
    assert (loader.data["sam_th_fits"] == 99.0).all()


def test_export_header_records_the_override(
    tmp_path, frame_builders, fits_writer, header_dir
):
    from pxr_reduce.dataset import ReducedDataset

    beam_image, frame_header = frame_builders
    directory, make = header_dir
    files, frames = [], []
    for i, th in enumerate([0.0, 0.0, 1.2263, 1.2789, 1.3316, 1.3842]):
        header = frame_header(99.0, 1.0 if i >= 2 else 0.0)
        files.append(
            fits_writer(tmp_path / f"MF999A_{i}.fits", beam_image(5000.0), header)
        )
        goals = dict(_GOALS, **{"Sample Theta": th, "CCD Theta": 2 * th,
                                "Sample Z": 1.0 if i >= 2 else 0.0})
        frames.append((f"MF999A_{i}.fits", goals, {"Sample Theta": 0.001}))
    make("MF999A.txt", frames)
    header_file.index_header_directory.cache_clear()

    loader = PXRLoader(
        files,
        ReductionConfig(roi_height=9, roi_width=9, header=directory,
                        sam_th_correction=False),
    )
    loader.process()
    header = "\n".join(ReducedDataset.from_loader(loader).header_lines())

    # A reader must be able to tell the metadata did not come from the FITS files.
    assert "header files, NOT the FITS headers" in header
    assert "frame(s) overridden" in header
    assert "q uses the readback" in header
