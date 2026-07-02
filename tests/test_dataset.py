import pandas as pd
import pytest

from pxr_reduce.dataset import ReducedDataset


@pytest.fixture
def dataset(processed_loader_factory):
    loader = processed_loader_factory()
    return ReducedDataset.from_loader(loader)


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
