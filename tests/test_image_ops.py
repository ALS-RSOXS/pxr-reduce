import numpy as np
import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.detectors import get_detector
from pxr_reduce.image_ops import (
    build_series_mask,
    clean_image,
    dark_roi_slices,
    dezinger,
    integrate_frame,
    locate_beam,
    roi_slices,
    trim,
)


def test_trim_removes_edges():
    img = np.arange(100, dtype=float).reshape(10, 10)
    out = trim(img, 2, 3)
    assert out.shape == (6, 4)


def test_clean_image_shape_after_trim():
    img = np.zeros((50, 50))
    cfg = ReductionConfig(trim_x=5, trim_y=5, filter_size=3)
    out = clean_image(img, cfg)
    assert out.shape == (40, 40)


def test_clean_image_skips_dezinger_when_disabled():
    # a lone hot pixel survives untouched when dezinger is off
    img = np.zeros((50, 50))
    img[25, 25] = 9999.0
    cfg = ReductionConfig(trim_x=5, trim_y=5, dezinger=False)
    out = clean_image(img, cfg)
    assert out.shape == (40, 40)
    assert out[20, 20] == 9999.0  # trimmed coords of (25,25)


def test_dezinger_threshold_controls_aggressiveness():
    # A pixel ~6x its local background: removed at threshold 5, kept at threshold 10.
    img = np.full((15, 15), 100.0)
    img[7, 7] = 600.0
    aggressive = dezinger(img, ReductionConfig(dezinger_threshold=5.0, filter_size=3))
    lenient = dezinger(img, ReductionConfig(dezinger_threshold=10.0, filter_size=3))
    assert aggressive[7, 7] < 600.0  # replaced by local median (~100)
    assert lenient[7, 7] == 600.0  # below 10x median -> kept


def test_build_series_mask_dilates_hot_region():
    # a single hot pixel at center; mean stays above threshold
    imgs = []
    for _ in range(3):
        a = np.zeros((41, 41))
        a[20, 20] = 1000.0
        imgs.append(a)
    cfg = ReductionConfig(mask_threshold=100, drift_distance=5)
    mask = build_series_mask(imgs, cfg)
    assert mask[20, 20]
    # dilated within radius 5
    assert mask[20, 24]
    assert mask[24, 20]
    # outside radius stays masked off
    assert not mask[20, 30]


def test_build_series_mask_empty_raises():
    with pytest.raises(ValueError):
        build_series_mask([], ReductionConfig())


def test_mask_bounding_box_pads_and_clips():
    from pxr_reduce.image_ops import mask_bounding_box

    mask = np.zeros((100, 100), dtype=bool)
    mask[40:50, 60:70] = True
    row_sl, col_sl = mask_bounding_box(mask, pad=5)
    assert row_sl == slice(35, 55)
    assert col_sl == slice(55, 75)
    # padding clips at the image edge
    mask2 = np.zeros((20, 20), dtype=bool)
    mask2[0:3, 0:3] = True
    r, c = mask_bounding_box(mask2, pad=10)
    assert r.start == 0 and c.start == 0


def test_mask_bounding_box_empty_returns_full_frame():
    from pxr_reduce.image_ops import mask_bounding_box

    mask = np.zeros((30, 40), dtype=bool)
    r, c = mask_bounding_box(mask, pad=5)
    assert r == slice(0, 30)
    assert c == slice(0, 40)


def test_locate_beam_finds_peak_in_mask():
    img = np.zeros((20, 20))
    img[5, 7] = 100.0
    img[15, 15] = 500.0  # brighter but outside mask
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:10, 0:10] = True
    assert locate_beam(img, mask) == (5, 7)


def test_roi_slices_centered():
    cfg = ReductionConfig(roi_height=4, roi_width=6)
    sly, slx = roi_slices((10, 10), cfg)
    assert sly == slice(8, 12)
    assert slx == slice(7, 13)


def test_dark_roi_uses_lhs_when_room():
    cfg = ReductionConfig(roi_height=4, roi_width=6, dark_pix_offset=5, darkside="LHS")
    _, slx = dark_roi_slices((10, 40), cfg)
    # x_low = 40 - 3 - 6 - 5 = 26
    assert slx.start == 26


def test_dark_roi_falls_back_to_rhs_when_no_room():
    cfg = ReductionConfig(roi_height=4, roi_width=6, dark_pix_offset=5, darkside="LHS")
    _, slx = dark_roi_slices((10, 3), cfg)  # too close to left edge
    # x_low = 3 + 3 + 5 = 11
    assert slx.start == 11


def test_integrate_frame_net_and_ratio():
    # pure-Poisson detector (gain 1, no read/dark/bias)
    det = get_detector("default")
    img = np.zeros((41, 41))
    img[20, 20] = 100.0  # beam
    img[20, 20 - 5] = 0.0  # dark region low
    mask = np.zeros((41, 41), dtype=bool)
    mask[15:25, 15:25] = True
    cfg = ReductionConfig(roi_height=3, roi_width=3, dark_pix_offset=2, darkside="LHS")
    result = integrate_frame(img, mask, cfg, det, exposure_s=1.0)
    assert result.beam_spot == (20, 20)
    assert result.counts_spot == pytest.approx(100.0)
    assert result.net.value == pytest.approx(result.counts_spot - result.counts_dark)
    # net uncertainty is sqrt(var_spot + var_dark) > 0
    assert result.net.sigma > 0
