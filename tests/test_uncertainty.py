import numpy as np
import pytest

from pxr_reduce.detectors import DetectorSpec
from pxr_reduce.uncertainty import (
    Value,
    apply_scale_factor,
    net_counts,
    product,
    ratio,
    roi_variance,
    scale,
)


@pytest.fixture
def poisson_detector():
    # gain=1, no read/dark/bias -> pure Poisson, variance == counts
    return DetectorSpec(name="poisson", description="", pixel_size_mm=0.1)


def test_value_rel():
    assert Value(10.0, 2.0).rel == pytest.approx(0.2)
    assert Value(0.0, 5.0).rel == 0.0


def test_roi_variance_sums_pixel_variance(poisson_detector):
    roi = np.array([[100.0, 100.0], [100.0, 100.0]])
    # pure Poisson: variance == sum of counts == 400
    assert roi_variance(roi, poisson_detector, 1.0) == pytest.approx(400.0)


def test_net_counts_subtracts_and_adds_variance(poisson_detector):
    spot = np.full((2, 2), 100.0)  # sum 400, var 400
    dark = np.full((2, 2), 25.0)  # sum 100, var 100
    result = net_counts(spot, dark, poisson_detector, 1.0)
    assert result.value == pytest.approx(300.0)
    assert result.sigma == pytest.approx(np.sqrt(500.0))


def test_scale_is_linear():
    v = scale(Value(10.0, 2.0), 0.5)
    assert v.value == pytest.approx(5.0)
    assert v.sigma == pytest.approx(1.0)


def test_ratio_propagates_relative_errors_in_quadrature():
    num = Value(100.0, 10.0)  # 10% rel
    den = Value(50.0, 5.0)  # 10% rel
    r = ratio(num, den)
    assert r.value == pytest.approx(2.0)
    assert r.rel == pytest.approx(np.sqrt(0.02))


def test_ratio_zero_denominator_returns_zero():
    r = ratio(Value(1.0, 0.1), Value(0.0, 0.0))
    assert r.value == 0.0
    assert r.sigma == 0.0


def test_product_propagates_relative_errors():
    p = product(Value(2.0, 0.2), Value(3.0, 0.3))  # both 10%
    assert p.value == pytest.approx(6.0)
    assert p.rel == pytest.approx(np.sqrt(0.02))


def test_apply_scale_factor_propagates_scale_uncertainty():
    r = Value(10.0, 1.0)  # 10%
    scale_factor = Value(2.0, 0.2)  # 10%
    scaled = apply_scale_factor(r, scale_factor)
    assert scaled.value == pytest.approx(5.0)
    # scale uncertainty must contribute (unlike the old commented-out code)
    assert scaled.rel == pytest.approx(np.sqrt(0.02))
    assert scaled.sigma > r.rel * scaled.value
