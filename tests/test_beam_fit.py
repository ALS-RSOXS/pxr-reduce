import numpy as np
import pytest

from pxr_reduce.beam_fit import (
    BeamShape,
    aggregate_shapes,
    estimate_moments,
    roi_from_shape,
)


def _gaussian(size, sigma_y, sigma_x, amp=1000.0, background=5.0):
    c = size // 2
    yy, xx = np.indices((size, size))
    g = amp * np.exp(-((yy - c) ** 2) / (2 * sigma_y**2) - ((xx - c) ** 2) / (2 * sigma_x**2))
    return g + background


def test_estimate_moments_recovers_sigma():
    img = _gaussian(81, sigma_y=4.0, sigma_x=6.0)
    shape = estimate_moments(img)
    assert shape.success
    assert shape.sigma_y == pytest.approx(4.0, abs=0.4)
    assert shape.sigma_x == pytest.approx(6.0, abs=0.4)
    assert shape.centroid_y == pytest.approx(40, abs=0.5)


def test_estimate_moments_no_signal_fails():
    shape = estimate_moments(np.full((10, 10), 7.0))  # flat: no signal after bg sub
    assert not shape.success


def test_aggregate_shapes_medians():
    shapes = [
        BeamShape(20, 20, 3.0, 4.0, 100, True),
        BeamShape(20, 20, 3.2, 4.4, 110, True),
        BeamShape(20, 20, 2.8, 3.6, 90, True),
        BeamShape(0, 0, np.nan, np.nan, 0, False),  # failed, ignored
    ]
    agg = aggregate_shapes(shapes)
    assert agg is not None
    assert agg.sigma_y == pytest.approx(3.0)
    assert agg.sigma_x == pytest.approx(4.0)


def test_aggregate_shapes_all_failed_returns_none():
    assert aggregate_shapes([BeamShape(0, 0, np.nan, np.nan, 0, False)]) is None


def test_roi_from_shape_spans_n_sigma():
    shape = BeamShape(0, 0, sigma_y=3.0, sigma_x=5.0, amplitude=1, success=True)
    h, w = roi_from_shape(shape, n_sigma=3.0)
    assert h == int(np.ceil(2 * 3 * 3.0))  # 18
    assert w == int(np.ceil(2 * 3 * 5.0))  # 30


def test_roi_from_shape_respects_minimum():
    shape = BeamShape(0, 0, sigma_y=0.1, sigma_x=0.1, amplitude=1, success=True)
    h, w = roi_from_shape(shape, n_sigma=1.0, minimum=5)
    assert h == 5 and w == 5
