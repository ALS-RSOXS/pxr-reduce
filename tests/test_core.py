import numpy as np
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader


def _frame_header(index, sam_th, sam_z, energy=250.0):
    """Build a full raw header for a synthetic frame."""
    return {
        "Beamline Energy": energy,
        "EPU Polarization": 100.0,
        "Sample Theta": sam_th,
        "CCD Theta": 2 * sam_th,
        "Sample X": 0.0,
        "Sample Y": 0.0,
        "Sample Z": sam_z,
        "EXPOSURE": 1.0,
        "Higher Order Suppressor": 5.0,
        "Upstream JJ Vert Aperture": 0.1,
        "Upstream JJ Horz Aperture": 0.1,
        "Beam Current": 500.0,
        "AI 3 Izero": 1.0,
    }


def _beam_image(peak, size=61, center=(30, 30)):
    """A small gaussian-ish beam blob on a low background."""
    img = np.full((size, size), 5.0)
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    img += peak * np.exp(-r2 / (2 * 3.0**2))
    return img


@pytest.fixture
def loader(tmp_path, fits_writer):
    # 2 direct-beam frames (sam_z=0), then a descending-intensity reflectivity scan.
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    files = []
    for i, (peak, z, th) in enumerate(zip(peaks, sam_z, sam_th)):
        path = tmp_path / f"MF999A_{i}.fits"
        fits_writer(path, _beam_image(peak), _frame_header(i, th, z))
        files.append(path)
    return PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))


def test_loader_infers_sample_name_and_count(loader):
    assert loader.name == "MF999A_"
    assert len(loader) == 6
    assert not loader.data_processed


def test_loader_metadata_has_derived_columns(loader):
    assert "q" in loader.data.columns
    assert "wavelength" in loader.data.columns
    assert loader.data["fits_index"].tolist() == [0, 1, 2, 3, 4, 5]


def test_get_image_and_clean_image(loader):
    raw = loader.get_image(0)
    clean = loader.get_clean_image(0)
    assert raw.shape == (61, 61)
    # cleaned image is trimmed
    assert clean.shape[0] < raw.shape[0]


def test_query_by_range(loader):
    subset = loader.query(sam_th=(1.0, 3.0))
    assert sorted(subset["sam_th"].tolist()) == [1.0, 2.0, 3.0]


def test_reduce_requires_process(loader):
    with pytest.raises(RuntimeError):
        loader.reduce()


def test_end_to_end_reduction(loader):
    loader.process()
    assert loader.data_processed
    assert loader.mask is not None
    # counts columns populated
    for col in ["counts_spot", "counts_dark", "counts_refl", "counts_err"]:
        assert col in loader.data.columns
    out = loader.reduce()
    assert {"sam_th", "q", "R", "R_err"}.issubset(out.columns)
    assert len(out) > 0
    assert (out["R"] > 0).all()
    assert (out["R_err"] >= 0).all()


def test_quick_mode_runs(loader):
    loader.process()
    quick = loader.reduce(apply_scale=False)
    assert len(quick) > 0


def test_subsample_helper():
    from pxr_reduce.core import _subsample

    assert _subsample([0, 1, 2, 3], 0) == [0, 1, 2, 3]  # 0 = all
    assert _subsample([0, 1, 2, 3], 10) == [0, 1, 2, 3]  # fewer than cap
    assert _subsample(list(range(10)), 3) == [0, 4, 8]  # capped (ceil step), <=3


def test_process_with_roi_from_beam_fit(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader

    beam_image, frame_header = frame_builders
    # 2 direct-beam frames (sam_z=0) then reflectivity frames.
    files = []
    for i, (z, th) in enumerate([(0, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4)]):
        p = tmp_path / f"FIT_{i}.fits"
        fits_writer(p, beam_image(10000.0), frame_header(th, z))
        files.append(p)
    loader = PXRLoader(files, ReductionConfig(roi_from_beam_fit=True, roi_n_sigma=3.0))
    loader.process()
    # a beam shape was fit and the ROI was sized from it
    assert loader.beam_shape is not None
    assert loader.beam_shape.success
    assert loader.config.roi_height > 0
    assert len(loader.reduce()) > 0


def test_roi_from_beam_fit_falls_back_without_i0(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader

    beam_image, frame_header = frame_builders
    # No direct-beam frames (sam_z always in beam) -> fit finds no i0, falls back.
    files = []
    for i, th in enumerate([1, 2, 3, 4]):
        p = tmp_path / f"NOI0_{i}.fits"
        fits_writer(p, beam_image(8000.0), frame_header(th, 1.0))
        files.append(p)
    cfg = ReductionConfig(roi_from_beam_fit=True, roi_height=12, roi_width=12)
    loader = PXRLoader(files, cfg)
    loader.process()
    assert loader.beam_shape is None
    assert loader.config.roi_height == 12  # unchanged fallback


def test_process_with_dezinger_disabled(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader

    beam_image, frame_header = frame_builders
    files = []
    for i, (z, th) in enumerate([(0, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4)]):
        p = tmp_path / f"FAST_{i}.fits"
        fits_writer(p, beam_image(5000.0), frame_header(th, z))
        files.append(p)
    loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9, dezinger=False))
    loader.process()
    assert loader.data_processed
    assert len(loader.reduce()) > 0
