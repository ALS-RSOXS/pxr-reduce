import math
from pathlib import Path

import pandas as pd
import pytest

from pxr_reduce.dataset import (
    ReducedDataset,
    _error_decimals,
    _q_uncertainty,
    _round_to_decimals,
    _round_value_and_error,
)
from pxr_reduce.provenance import ReductionProvenance


@pytest.fixture
def dataset(processed_loader_factory):
    loader = processed_loader_factory()
    return ReducedDataset.from_loader(loader)


def _make_dataset(data: pd.DataFrame) -> ReducedDataset:
    """Build a minimal dataset (no sources) for export-formatting tests."""
    prov = ReductionProvenance.create([], reduction_time=None, cwd=Path("."))
    return ReducedDataset(data=data, provenance=prov)


def test_from_loader_captures_provenance(dataset):
    prov = dataset.provenance
    assert len(prov.sources) == 1
    src = prov.sources[0]
    assert src.sample_name == "MF999A_"
    assert src.n_frames == 6
    assert 250.0 in src.energies
    assert prov.reduction_time  # non-empty ISO string
    assert prov.uncertainty_model


def test_header_lines_contain_key_provenance(dataset):
    header = "\n".join(dataset.header_lines())
    assert "pxr-reduce reduced dataset" in header
    assert "Reduction time" in header
    assert "Source 1" in header
    assert "PLACEHOLDER noise specs" in header  # default detector is placeholder
    assert "Columns:" in header


def test_save_dat_writes_and_roundtrips(dataset, tmp_path):
    out = dataset.save_dat(tmp_path / "MF999A")
    assert out.exists()
    assert out.suffix == ".dat"
    text = out.read_text()
    assert text.startswith("# pxr-reduce reduced dataset")
    # data round-trips ignoring comment lines
    df = pd.read_csv(out, sep="\t", comment="#")
    assert {"q", "R", "R_err"}.issubset(df.columns)
    assert len(df) == len(dataset.data)


def test_save_dat_dry_run_writes_nothing(dataset, tmp_path):
    out = dataset.save_dat(tmp_path / "dry", dry_run=True)
    assert not out.exists()


def test_save_plots_one_png_per_group(dataset, tmp_path):
    paths = dataset.save_plots(tmp_path / "plots")
    assert len(paths) == 1  # single energy/polarization
    assert all(p.exists() for p in paths)
    assert paths[0].suffix == ".png"


def test_save_creates_dat_and_plots_folder(dataset, tmp_path):
    result = dataset.save(tmp_path / "MF999A")
    assert result["dat"].exists()
    assert (tmp_path / "MF999A_plots").is_dir()
    assert all(p.exists() for p in result["plots"])


def test_save_dry_run_writes_nothing(dataset, tmp_path):
    result = dataset.save(tmp_path / "MF999A", dry_run=True)
    assert not result["dat"].exists()
    assert not (tmp_path / "MF999A_plots").exists()


def test_combine_merges_data_and_sources(processed_loader_factory):
    spol = ReducedDataset.from_loader(
        processed_loader_factory(sample="MF999A_spol", polarization=100.0, subdir="s")
    )
    ppol = ReducedDataset.from_loader(
        processed_loader_factory(sample="MF999A_ppol", polarization=190.0, subdir="p")
    )
    combined = spol.combine(ppol)
    assert len(combined.provenance.sources) == 2
    assert len(combined.data) == len(spol.data) + len(ppol.data)
    header = "\n".join(combined.header_lines())
    assert "Source 1" in header
    assert "Source 2" in header


def test_combine_all_requires_nonempty():
    with pytest.raises(ValueError):
        ReducedDataset.combine_all([])


# -- Export rounding / formatting --------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        (0.0003925, 4),  # PDG: "392" -> 1 sig fig
        (0.0001388, 5),  # PDG: "138" -> 2 sig figs
        (0.012, 3),  # PDG: "120" -> 2 sig figs
        (12.0, 0),  # 2 sig figs at the tens place
        (0.0, None),  # unusable
        (-1.0, None),  # unusable
        (float("nan"), None),  # unusable
        (float("inf"), None),  # unusable
    ],
)
def test_error_decimals_follows_pdg_rule(error, expected):
    assert _error_decimals(error) == expected


def test_round_value_and_error_share_precision():
    value, error = _round_value_and_error(0.012483685, 0.0003925)
    assert value == "0.0125"
    assert error == "0.0004"


def test_round_to_decimals_handles_no_uncertainty():
    # None -> general 6-figure precision, no float display noise.
    assert _round_to_decimals(0.1 + 0.2, None) == "0.3"


def test_q_uncertainty_undefined_at_zero_theta():
    assert math.isnan(_q_uncertainty(0.01, 0.0, 4))


def test_q_uncertainty_scales_with_angle_step():
    coarse = _q_uncertainty(0.0074, 1.05, 3)
    fine = _q_uncertainty(0.0074, 1.05, 4)
    assert coarse == pytest.approx(fine * 10)


def test_save_dat_has_clean_line_endings(tmp_path):
    data = pd.DataFrame(
        {
            "scan": [0.0, 0.0],
            "energy": [250.0, 250.0],
            "polarization": [100.0, 100.0],
            "sam_th": [1.05, 1.10],
            "q": [0.0074, 0.0075],
            "R": [0.0125, 0.0698],
            "R_err": [0.0004, 0.0006],
        }
    )
    out = _make_dataset(data).save_dat(tmp_path / "endings")
    raw = out.read_bytes()
    assert b"\r\r\n" not in raw  # no stray CR from double newline translation
    assert b"\r" not in raw  # clean LF-only output
    # No blank line between data rows: body line count matches header + rows.
    assert raw.count(b"\n") == len(out.read_text().splitlines())


def test_save_dat_rounds_columns(tmp_path):
    data = pd.DataFrame(
        {
            "scan": [2.0],
            "energy": [399.0],
            "polarization": [100.0],
            "sam_th": [1.1011000000000002],
            "q": [0.007418454964923283],
            "R": [0.012483685807533165],
            "R_err": [0.0003925529419735068],
        }
    )
    out = _make_dataset(data).save_dat(tmp_path / "rounded")
    df = pd.read_csv(out, sep="\t", comment="#", dtype=str)
    row = df.iloc[0]
    assert row["R"] == "0.0125"
    assert row["R_err"] == "0.0004"
    assert row["sam_th"] == "1.1011"  # 4 decimals, float noise gone
    assert row["q"] == "0.0074185"  # precision tied to angular step


def test_export_rounding_leaves_source_data_untouched(tmp_path):
    data = pd.DataFrame(
        {
            "scan": [0.0],
            "energy": [250.0],
            "polarization": [100.0],
            "sam_th": [1.1011000000000002],
            "q": [0.007418454964923283],
            "R": [0.012483685807533165],
            "R_err": [0.0003925529419735068],
        }
    )
    ds = _make_dataset(data)
    ds.save_dat(tmp_path / "untouched")
    # Full precision is preserved in the in-memory table (plots use it).
    assert ds.data["R"].iloc[0] == 0.012483685807533165
