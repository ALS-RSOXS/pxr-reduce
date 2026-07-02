import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.detectors import DetectorSpec


def test_defaults():
    cfg = ReductionConfig()
    assert cfg.exposure_offset == 0.00389278
    assert cfg.energy_resolution == 20.0
    assert cfg.roi_height == 40
    assert cfg.darkside == "LHS"
    assert cfg.dezinger is True


def test_detector_spec_resolves_by_name():
    cfg = ReductionConfig(detector="default")
    assert cfg.detector_spec().name == "default"


def test_detector_spec_accepts_instance():
    spec = DetectorSpec(name="custom", description="", pixel_size_mm=0.05)
    cfg = ReductionConfig(detector=spec)
    assert cfg.detector_spec() is spec


def test_invalid_darkside_raises():
    with pytest.raises(ValueError):
        ReductionConfig(darkside="MIDDLE")  # type: ignore[arg-type]


def test_invalid_energy_resolution_raises():
    with pytest.raises(ValueError):
        ReductionConfig(energy_resolution=0)


def test_invalid_roi_raises():
    with pytest.raises(ValueError):
        ReductionConfig(roi_width=0)


def test_to_header_dict_expands_detector_and_drops_reference():
    cfg = ReductionConfig(detector="default")
    header = cfg.to_header_dict()
    assert "detector" not in header
    assert header["detector_name"] == "default"
    assert header["roi_height"] == 40
    assert header["energy_resolution"] == 20.0
