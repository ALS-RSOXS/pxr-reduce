import numpy as np
import pytest

from pxr_reduce.detectors import (
    DetectorSpec,
    available_detectors,
    get_detector,
    register_detector,
    with_noise,
)


def test_default_detector_is_registered():
    assert "default" in available_detectors()
    det = get_detector("default")
    assert det.pixel_size_mm == 0.027
    assert det.bit_depth == 16


def test_saturation_adu_matches_bit_depth():
    det = get_detector("default")
    assert det.saturation_adu == 2**16 - 1


def test_get_detector_passthrough_of_spec():
    spec = DetectorSpec(name="x", description="d", pixel_size_mm=0.05)
    assert get_detector(spec) is spec


def test_get_detector_unknown_name_raises():
    with pytest.raises(KeyError):
        get_detector("does_not_exist")


def test_default_noise_is_flagged_as_placeholder():
    assert get_detector("default").has_measured_noise is False


def test_with_noise_produces_measured_variant_without_mutating_base():
    base = get_detector("default")
    measured = with_noise(
        base, name="measured", gain_e_per_adu=2.0, read_noise_adu=3.0
    )
    assert measured.has_measured_noise is True
    assert measured.gain_e_per_adu == 2.0
    assert measured.read_noise_adu == 3.0
    # base unchanged (frozen/immutable)
    assert base.has_measured_noise is False
    assert base.gain_e_per_adu == 1.0


def test_pixel_variance_pure_poisson_when_no_extra_noise():
    # gain=1, no read/dark/bias -> variance == signal (Poisson)
    det = DetectorSpec(name="p", description="", pixel_size_mm=0.1)
    signal = np.array([[100.0, 400.0], [0.0, 900.0]])
    var = det.pixel_variance_adu(signal, exposure_s=1.0)
    np.testing.assert_allclose(var, signal)


def test_pixel_variance_includes_read_and_dark_terms():
    det = DetectorSpec(
        name="rn",
        description="",
        pixel_size_mm=0.1,
        gain_e_per_adu=1.0,
        read_noise_adu=5.0,
        dark_current_adu_per_s=2.0,
    )
    signal = np.array([[100.0]])
    var = det.pixel_variance_adu(signal, exposure_s=3.0)
    # 100 (shot) + 25 (read^2) + 6 (dark*t)
    np.testing.assert_allclose(var, [[131.0]])


def test_pixel_variance_clips_negative_signal():
    det = DetectorSpec(name="b", description="", pixel_size_mm=0.1, bias_adu=50.0)
    signal = np.array([[10.0]])  # below bias
    var = det.pixel_variance_adu(signal, exposure_s=1.0)
    assert var[0, 0] == 0.0


def test_is_saturated():
    det = get_detector("default")
    hot = np.full((3, 3), 2**16 - 1, dtype=float)
    cool = np.zeros((3, 3))
    assert det.is_saturated(hot, threshold=2.0) is True
    assert det.is_saturated(cool, threshold=2.0) is False


def test_register_detector_rejects_duplicate_without_overwrite():
    spec = DetectorSpec(name="dup", description="", pixel_size_mm=0.1)
    register_detector(spec)
    with pytest.raises(ValueError):
        register_detector(spec)
    # overwrite allowed
    register_detector(spec, overwrite=True)


def test_to_header_dict_is_flat_and_serializable():
    header = get_detector("default").to_header_dict()
    assert header["detector_name"] == "default"
    assert header["detector_saturation_adu"] == 2**16 - 1
    assert all(not isinstance(v, dict) for v in header.values())
