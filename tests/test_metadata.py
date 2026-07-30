import numpy as np
import pandas as pd
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.metadata import (
    HEADER_NAMES,
    apply_energy_and_theta,
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


def test_label_scans_always_splits_on_a_scan_id_change():
    # The angle jump here is only 1 deg, far below new_scan_marker, so the angle rule
    # alone would merge the two scans into one sweep -- they would then share an I0 and
    # be stitched to each other.
    df = pd.DataFrame(
        {
            "scan_id": [11, 11, 11, 22, 22, 22],
            "sam_th": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    out = label_scans(df, ReductionConfig(new_scan_marker=15.0))
    assert out["scan"].tolist() == [0, 0, 0, 1, 1, 1]
    # sweep is a 0-based ordinal *within* each scan, so it is stable regardless of
    # what else was pooled.
    assert out["sweep"].tolist() == [0, 0, 0, 0, 0, 0]


def test_label_scans_numbers_sweeps_within_each_scan():
    df = pd.DataFrame(
        {
            "scan_id": [11, 11, 11, 11, 22, 22],
            # Scan 11 sweeps 1->2 then jumps back to start a second sweep at 60->61;
            # scan 22 then starts its own sweep 0.
            "sam_th": [1.0, 2.0, 60.0, 61.0, 1.0, 2.0],
        }
    )
    out = label_scans(df, ReductionConfig(new_scan_marker=15.0))
    assert out["scan"].tolist() == [0, 0, 1, 1, 2, 2]
    assert out["sweep"].tolist() == [0, 0, 1, 1, 0, 0]


def test_label_scans_without_scan_ids_falls_back_to_angle_jumps():
    df = pd.DataFrame({"sam_th": [1.0, 2.0, 60.0, 61.0]})
    out = label_scans(df, ReductionConfig(new_scan_marker=15.0))
    assert out["scan"].tolist() == [0, 0, 1, 1]
    assert out["sweep"].tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "scan_id,sweep,expected",
    [
        (2045, 0, "id2045_sweep0_E283.5_P100"),
        (2045.0, 3.0, "id2045_sweep3_E283.5_P100"),  # floats from a mixed-dtype row
    ],
)
def test_sweep_tag_formats_identifiers_as_integers(scan_id, sweep, expected):
    from pxr_reduce.metadata import sweep_tag

    assert sweep_tag(scan_id, sweep, 283.5, 100.0) == expected


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


def test_determine_sam_th_offset_tolerates_a_move_on_the_final_frame():
    # A sam_z move on the last frame has no "frame after" it. Only the first move
    # anchors the geometry, so this must not run past the end of the table.
    df = pd.DataFrame(
        {
            "sam_z": [0.0, 1.0, 1.0, 2.0],  # moves at index 1 *and* index 3
            "sam_th": [0.0, 0.0, 3.0, 4.0],
            "det_th": [0.0, 0.0, 8.0, 10.0],
        }
    )
    assert determine_sam_th_offset(df) == pytest.approx(1.0)


def test_determine_sam_th_offset_raises_without_a_usable_move():
    df = pd.DataFrame(
        {"sam_z": [1.0, 1.0, 1.0], "sam_th": [0.0, 1.0, 2.0], "det_th": [0.0, 2.0, 4.0]}
    )
    with pytest.raises(IndexError, match="No sam_z movement"):
        determine_sam_th_offset(df)


def test_optional_monitor_headers_are_substituted(caplog):
    # Older beamline configurations do not record these; the reduction must proceed.
    records = [_raw_record(i) for i in range(3)]
    for record in records:
        del record["Beam Current"]
        del record["AI 3 Izero"]
    with caplog.at_level("WARNING"):
        out = build_metadata_table(records, ReductionConfig())
    assert out["beam_current"].tolist() == [1.0, 1.0, 1.0]
    assert out["i0"].tolist() == [1.0, 1.0, 1.0]
    assert "do not record 'Beam Current'" in caplog.text
    assert "decay across the scan is not corrected" in caplog.text


def test_missing_required_header_still_raises():
    records = [_raw_record(i) for i in range(2)]
    for record in records:
        del record["Sample Theta"]
    with pytest.raises(KeyError, match="Sample Theta"):
        build_metadata_table(records, ReductionConfig())


def test_prepare_metadata_adds_q_and_wavelength():
    records = [_raw_record(i, sam_th=float(i)) for i in range(4)]
    # give a sam_z move so offset determination works
    records[1]["Sample Z"] = 1.0
    records[2]["Sample Z"] = 1.0
    records[3]["Sample Z"] = 1.0
    df = build_metadata_table(records, ReductionConfig())
    out, offsets = prepare_metadata(df, ReductionConfig())
    assert "q" in out.columns
    assert "wavelength" in out.columns
    assert np.all(out["wavelength"] > 0)
    assert all(isinstance(value, float) for value in offsets.values())


def test_direct_beam_mask_per_scan():
    df = pd.DataFrame(
        {
            "scan": [0, 0, 0, 0, 1, 1, 1],
            "sam_z": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    # Direct beam is strictly before the sam_z move (same as normalization):
    # scan 0 moves at pos 2 -> frames 0,1 direct
    # scan 1 moves at pos 1 -> frame 0 direct
    mask = direct_beam_mask(df)
    assert mask.tolist() == [True, True, False, False, True, False, False]


def test_direct_beam_mask_no_move_marks_none():
    df = pd.DataFrame({"scan": [0, 0, 0], "sam_z": [1.0, 1.0, 1.0]})
    assert direct_beam_mask(df).tolist() == [False, False, False]


def test_prepare_metadata_uses_explicit_offset():
    records = [_raw_record(i, sam_th=1.0) for i in range(3)]
    df = build_metadata_table(records, ReductionConfig())
    cfg = ReductionConfig(sam_th_offset=0.5)
    out, offsets = prepare_metadata(df, cfg)
    # The offset is now reported per scan; an explicit one applies to every scan.
    assert set(offsets.values()) == {0.5}
    assert out["sam_th"].iloc[0] == pytest.approx(1.5)


def test_offset_is_determined_per_scan_id():
    # Two scans pooled into one sample, aligned differently: scan 11 sits at
    # det_th/2 - sam_th = 1.0 - 1.0 = 0.0, scan 22 at 1.0 - 0.8 = +0.2.
    df = pd.DataFrame(
        {
            "fits_index": list(range(8)),
            "scan": [0] * 8,
            "scan_id": [11] * 4 + [22] * 4,
            "sam_z": [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "sam_th": [0.0, 1.0, 2.0, 3.0, 0.0, 0.8, 1.8, 2.8],
            "det_th": [0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0],
            "energy": [250.0] * 8,
        }
    )
    out, offsets = apply_energy_and_theta(df, ReductionConfig())

    assert offsets == {11: pytest.approx(0.0), 22: pytest.approx(0.2)}
    # Each scan is shifted by its own offset, not by the first scan's.
    assert out["sam_th"].tolist()[1:4] == pytest.approx([1.0, 2.0, 3.0])
    assert out["sam_th"].tolist()[5:8] == pytest.approx([1.0, 2.0, 3.0])


def test_offset_falls_back_to_the_sweep_label_without_scan_ids():
    df = pd.DataFrame(
        {
            "fits_index": list(range(4)),
            "scan": [0, 0, 1, 1],
            "scan_id": [-1, -1, -1, -1],  # unresolved
            "sam_z": [0.0, 1.0, 0.0, 1.0],
            "sam_th": [0.0, 1.0, 0.0, 0.8],
            "det_th": [0.0, 2.0, 0.0, 2.0],
            "energy": [250.0] * 4,
        }
    )
    _, offsets = apply_energy_and_theta(df, ReductionConfig())
    assert set(offsets) == {0, 1}
