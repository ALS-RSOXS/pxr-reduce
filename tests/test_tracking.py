import numpy as np

from pxr_reduce.tracking import (
    LocateResult,
    anchor_position,
    locate_in_window,
    smooth_track,
)


def _blob(shape, center, amp=1000.0, sigma=2.0, background=5.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    g = amp * np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2))
    return background + g


def test_locate_in_window_finds_centroid():
    img = _blob((60, 60), (30, 33))
    res = locate_in_window(img, center=(30, 33), radius=8, snr_min=3.0)
    assert res.found
    assert abs(res.y - 30) <= 1
    assert abs(res.x - 33) <= 1
    assert res.snr > 3.0


def test_locate_in_window_ignores_far_noise_spike():
    img = _blob((80, 80), (40, 40), amp=200.0)
    img[10, 70] = 60000.0  # bright hot pixel far from the beam
    # searching near the beam must not jump to the far spike
    res = locate_in_window(img, center=(40, 40), radius=10, snr_min=3.0)
    assert res.found
    assert abs(res.y - 40) <= 2
    assert abs(res.x - 40) <= 2


def test_locate_in_window_centroid_ignores_scatter_beyond_centroid_radius():
    # Main beam at (40, 40); dimmer scatter at (40, 62) that still exceeds
    # centroid_frac*peak and lies within the search radius but beyond the
    # centroid window. The centroid must stay on the beam.
    img = _blob((80, 80), (40, 40), amp=1000.0)
    img += _blob((80, 80), (40, 62), amp=350.0, background=0.0)
    res = locate_in_window(
        img, center=(40, 40), radius=30, snr_min=3.0, centroid_radius=10
    )
    assert res.found
    assert abs(res.x - 40) <= 1  # not dragged toward the scatter at x=62
    assert abs(res.y - 40) <= 1


def test_locate_in_window_reports_dropout_when_faint():
    img = np.full((40, 40), 100.0) + np.random.default_rng(0).normal(0, 5, (40, 40))
    res = locate_in_window(img, center=(20, 20), radius=8, snr_min=8.0)
    assert not res.found
    # returns the search center when not found
    assert (res.y, res.x) == (20, 20)


def test_anchor_position_locates_bright_beam():
    img = _blob((100, 100), (60, 25), amp=5000.0)
    y, x = anchor_position(img)
    assert abs(y - 60) <= 1
    assert abs(x - 25) <= 1


def test_smooth_track_interpolates_dropouts():
    # a smoothly moving beam with two dropouts marked as not-found
    positions = [(10, 10), (11, 12), (0, 0), (13, 16), (14, 18), (0, 0), (16, 22)]
    found = [True, True, False, True, True, False, True]
    out = smooth_track(positions, found, poly_order=2)
    assert len(out) == 7
    # dropouts were replaced
    assert out[2][1] is True
    assert out[5][1] is True
    # and filled with values along the trajectory (monotonic-ish), not (0,0)
    assert out[2][0] != (0, 0)
    assert 11 <= out[2][0][0] <= 13
    # detected inliers are kept unchanged
    assert out[0] == ((10, 10), False)


def test_smooth_track_corrects_outlier():
    # a long, clean line with one large jump at index 5
    positions = [(10 + i, 20 + 2 * i) for i in range(11)]
    positions[5] = (200, 400)
    found = [True] * 11
    out = smooth_track(positions, found, poly_order=1)
    # the outlier should be flagged and pulled back toward the line (~15)
    assert out[5][1] is True
    assert out[5][0][0] < 100


def test_smooth_track_too_few_detections_holds():
    positions = [(5, 5), (0, 0), (0, 0)]
    found = [True, False, False]
    out = smooth_track(positions, found)
    assert out[0] == ((5, 5), False)


def test_locate_result_dataclass():
    r = LocateResult(True, 1, 2, 5.0)
    assert (r.found, r.y, r.x, r.snr) == (True, 1, 2, 5.0)
