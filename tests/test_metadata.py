import numpy as np
import pandas as pd
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.metadata import (
    HEADER_NAMES,
    build_metadata_table,
    clean_monitors,
    determine_sam_th_offset,
    direct_beam_mask,
    label_scans,
    prepare_metadata,
)


def _raw_record(index: int, **overrides):
    """Build a raw header record with all required keys, allowing overrides."""
    record = {
        "fits_index": index,
        "Beamline Energy": 250.0,
        "EPU Polarization": 100.0,
        "Sample Theta": 1.0,
        "CCD Theta": 2.0,
        "Sample X": 0.0,
        "Sample Y": 0.0,
        "Sample Z": 0.0,
        "EXPOSURE": 1.0,
        "Higher Order Suppressor": 5.0,
        "Upstream JJ Vert Aperture": 0.1,
        "Upstream JJ Horz Aperture": 0.1,
        "Beam Current": 500.0,
        "AI 3 Izero": 1.0,
    }
    # Map override standardized keys back through the raw names.
    inverse = {v: k for k, v in HEADER_NAMES.items()}
    for key, value in overrides.items():
        raw_key = inverse.get(key, key)
        record[raw_key] = value
    return record


def test_build_metadata_table_renames_and_sorts():
    records = [_raw_record(2), _raw_record(0), _raw_record(1)]
    df = build_metadata_table(records, ReductionConfig())
    assert list(df["fits_index"]) == [0, 1, 2]
    assert "energy" in df.columns
    assert "scan" in df.columns
    assert "Beamline Energy" not in df.columns


def test_build_metadata_table_missing_key_raises():
    bad = _raw_record(0)
    del bad["Beamline Energy"]
    with pytest.raises(KeyError):
        build_metadata_table([bad], ReductionConfig())


def test_energy_rounding_to_resolution():
    # resolution 20 -> nearest 0.05 eV
    records = [_raw_record(0, energy=250.03), _raw_record(1, energy=250.06)]
    df = build_metadata_table(records, ReductionConfig(energy_resolution=20))
    assert df["energy"].iloc[0] == pytest.approx(250.05)
    assert df["energy"].iloc[1] == pytest.approx(250.05)


def test_clean_monitors_replaces_bad_values():
    df = pd.DataFrame(
        {
            "beam_current": [10.0, 500.0],  # 10 < 50 -> 1
            "i0": [-1.0, 2.0],  # negative -> 1
            "exposure": [0.0, 2.0],  # 0 -> 1, else + offset
        }
    )
    cfg = ReductionConfig(exposure_offset=0.1)
    out = clean_monitors(df, cfg)
    assert out["beam_current"].tolist() == [1.0, 500.0]
    assert out["i0"].tolist() == [1.0, 2.0]
    assert out["exposure"].iloc[0] == 1.0
    assert out["exposure"].iloc[1] == pytest.approx(2.1)


def test_label_scans_increments_on_large_theta_jump():
    df = pd.DataFrame({"sam_th": [0.0, 1.0, 2.0, 100.0, 101.0]})
    out = label_scans(df, ReductionConfig(new_scan_marker=15))
    assert out["scan"].tolist() == [0, 0, 0, 1, 1]


def test_determine_sam_th_offset_theta_2theta():
    # sam_z moves at index 1 -> anchor frame is index 2
    df = pd.DataFrame(
        {
            "sam_z": [0.0, 1.0, 1.0, 1.0],
            "sam_th": [0.0, 0.0, 3.0, 4.0],
            "det_th": [0.0, 0.0, 8.0, 10.0],
        }
    )
    # anchor at index 2: det_th/2 - sam_th = 8/2 - 3 = 1.0
    assert determine_sam_th_offset(df) == pytest.approx(1.0)


def test_prepare_metadata_adds_q_and_wavelength():
    records = [_raw_record(i, sam_th=float(i)) for i in range(4)]
    # give a sam_z move so offset determination works
    records[1]["Sample Z"] = 1.0
    records[2]["Sample Z"] = 1.0
    records[3]["Sample Z"] = 1.0
    df = build_metadata_table(records, ReductionConfig())
    out, offset = prepare_metadata(df, ReductionConfig())
    assert "q" in out.columns
    assert "wavelength" in out.columns
    assert np.all(out["wavelength"] > 0)
    assert isinstance(offset, float)


def test_direct_beam_mask_per_scan():
    df = pd.DataFrame(
        {
            "scan": [0, 0, 0, 0, 1, 1, 1],
            "sam_z": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    # Same i0 cutoff as normalization (index < move_position + 1):
    # scan 0 moves at pos 2 -> cutoff 3 -> first 3 direct beam
    # scan 1 moves at pos 1 -> cutoff 2 -> first 2 direct beam
    mask = direct_beam_mask(df)
    assert mask.tolist() == [True, True, True, False, True, True, False]


def test_direct_beam_mask_no_move_marks_none():
    df = pd.DataFrame({"scan": [0, 0, 0], "sam_z": [1.0, 1.0, 1.0]})
    assert direct_beam_mask(df).tolist() == [False, False, False]


def test_prepare_metadata_uses_explicit_offset():
    records = [_raw_record(i, sam_th=1.0) for i in range(3)]
    df = build_metadata_table(records, ReductionConfig())
    cfg = ReductionConfig(sam_th_offset=0.5)
    out, offset = prepare_metadata(df, cfg)
    assert offset == 0.5
    assert out["sam_th"].iloc[0] == pytest.approx(1.5)
