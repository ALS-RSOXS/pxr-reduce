"""Tests for the simple median-filter + local-argmax beam tracker plug-in."""

from __future__ import annotations

import numpy as np
import pytest

import pxr_reduce.simple_track as st
from pxr_reduce.simple_track import (
    SimplePXRLoader,
    locate_peak,
    locate_peak_near,
    peak,
    peak_near,
    track,
)


def _spot(
    size: int,
    center: tuple[int, int],
    amplitude: float = 1000.0,
    bg: float = 5.0,
    sigma: float = 3.0,
) -> np.ndarray:
    """A gaussian beam blob on a low flat background."""
    img = np.full((size, size), bg)
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    img += amplitude * np.exp(-r2 / (2 * sigma**2))
    return img


# -- Pure primitives ----------------------------------------------------------


def test_locate_peak_finds_bright_spot():
    y, x = locate_peak(_spot(41, (25, 15)), filter_size=5)
    assert (abs(y - 25), abs(x - 15)) <= (1, 1)


def test_locate_peak_ignores_single_hot_pixel():
    img = _spot(41, (20, 20), amplitude=500.0)
    img[5, 35] = 60000.0  # lone zinger, far brighter than the beam
    y, x = locate_peak(img, filter_size=3)
    # Median filter rejects the isolated pixel, so the beam still wins.
    assert abs(y - 20) <= 1 and abs(x - 20) <= 1


def test_peak_and_peak_near_on_presmoothed_image():
    # Operate on an already-"smoothed" array (no filtering inside these helpers).
    img = _spot(61, (30, 20), amplitude=800.0)
    img += _spot(61, (30, 45), amplitude=2000.0) - 5.0  # brighter spot elsewhere
    assert peak(img) == pytest.approx((30, 45), abs=1)  # global peak
    assert peak_near(img, (30, 20), radius=8) == pytest.approx((30, 20), abs=1)


def test_locate_peak_near_stays_in_window():
    img = _spot(61, (30, 20), amplitude=800.0)
    img += _spot(61, (30, 45), amplitude=2000.0) - 5.0  # brighter spot outside window
    y, x = locate_peak_near(img, (30, 20), radius=8, filter_size=5)
    assert abs(y - 30) <= 1 and abs(x - 20) <= 1


def test_locate_peak_near_handles_edge_without_crashing():
    img = _spot(41, (2, 2), amplitude=800.0)
    y, x = locate_peak_near(img, (1, 1), radius=6, filter_size=3)
    assert y <= 6 and x <= 6


def test_locate_peak_near_empty_window_returns_center():
    img = _spot(41, (20, 20))
    assert locate_peak_near(img, (200, 200), radius=3, filter_size=3) == (200, 200)


# -- Trajectory ---------------------------------------------------------------


def test_track_follows_moving_spot():
    frames = [_spot(61, (30, 20 + 3 * i)) for i in range(6)]
    positions = track(frames, radius=6, filter_size=5)
    for i, (y, x) in enumerate(positions):
        assert abs(y - 30) <= 1 and abs(x - (20 + 3 * i)) <= 1


def test_track_reseed_recovers_after_jump():
    centers = [(30, 20), (30, 22), (30, 48), (30, 50)]  # big jump at index 2
    frames = [_spot(61, c) for c in centers]

    # Without a re-seed, the local window around (30, 22) cannot reach (30, 48).
    no_reseed = track(frames, radius=6, filter_size=5)
    assert abs(no_reseed[2][1] - 48) > 6

    # Re-seeding at index 2 recovers the jump via a global peak search.
    reseeded = track(frames, radius=6, filter_size=5, reseed_indices={2})
    assert abs(reseeded[2][1] - 48) <= 1
    assert abs(reseeded[3][1] - 50) <= 1


# -- SimplePXRLoader integration ---------------------------------------------


def test_simple_loader_seeds_reseeds_and_tracks(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig

    beam_image, frame_header = frame_builders

    # Direct beam at (30, 30); the specular jumps to (30, 48) after sam_z moves,
    # then drifts +2 px per frame. trim=2 keeps everything inside the frame.
    centers = [(30, 30), (30, 30), (30, 48), (30, 50), (30, 52), (30, 54)]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]

    files = []
    for i, (center, z, th, pk) in enumerate(zip(centers, sam_z, sam_th, peaks)):
        path = tmp_path / f"SIM_{i}.fits"
        fits_writer(path, beam_image(pk, center=center), frame_header(th, z))
        files.append(path)

    config = ReductionConfig(trim_x=2, trim_y=2, roi_height=9, roi_width=9,
                             dark_pix_offset=5)
    loader = SimplePXRLoader(files, config)
    # search_radius (6) is smaller than the 18 px jump, so the re-seed must carry
    # the transition; the small per-frame drift is handled by the local search.
    loader.process(search_radius=6, progress=False)

    assert loader.data_processed
    spots = {int(r.fits_index): tuple(r.beam_spot) for r in loader.data.itertuples()}
    # Beam positions are reported in trimmed coordinates (image coord - trim=2).
    assert spots[0] == pytest.approx((28, 28), abs=1)  # direct-beam seed
    assert spots[2] == pytest.approx((28, 46), abs=1)  # re-seeded first reflection
    assert spots[3] == pytest.approx((28, 48), abs=1)  # tracked locally
    assert spots[5] == pytest.approx((28, 52), abs=1)


def test_simple_loader_reduces_end_to_end(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig

    beam_image, frame_header = frame_builders
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]

    files = []
    for i, (z, th, pk) in enumerate(zip(sam_z, sam_th, peaks)):
        path = tmp_path / f"SIM_{i}.fits"
        fits_writer(path, beam_image(pk), frame_header(th, z))
        files.append(path)

    loader = SimplePXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
    loader.process(progress=False)
    reduced = loader.reduce()
    assert not reduced.empty
    assert {"q", "R", "R_err"}.issubset(reduced.columns)


def test_cropped_tracking_matches_full_frame(tmp_path, fits_writer, frame_builders):
    from pxr_reduce.config import ReductionConfig

    beam_image, frame_header = frame_builders
    # i0 then a specular beam drifting 1 px/frame — well within a small radius.
    centers = [(28, 28), (28, 28), (28, 30), (28, 31), (28, 32), (28, 33)]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]

    files = []
    for i, (c, z, th, pk) in enumerate(zip(centers, sam_z, sam_th, peaks)):
        path = tmp_path / f"SIM_{i}.fits"
        fits_writer(path, beam_image(pk, center=c), frame_header(th, z))
        files.append(path)

    cfg = dict(trim_x=2, trim_y=2, roi_height=5, roi_width=5, dark_pix_offset=3)
    cropped = SimplePXRLoader(files, ReductionConfig(**cfg))
    cropped.process(search_radius=4, progress=False)  # small -> real crop
    full = SimplePXRLoader(files, ReductionConfig(**cfg))
    full.process(search_radius=100, progress=False)  # huge -> whole frame

    # Cropping the median filter must not change the beam or the integrated counts.
    assert list(cropped.data["beam_spot"]) == list(full.data["beam_spot"])
    np.testing.assert_array_equal(
        cropped.data["counts_spot"].to_numpy(), full.data["counts_spot"].to_numpy()
    )
    np.testing.assert_array_equal(
        cropped.data["counts_dark"].to_numpy(), full.data["counts_dark"].to_numpy()
    )


def test_simple_loader_median_filters_once_per_frame(
    tmp_path, fits_writer, frame_builders, monkeypatch
):
    from pxr_reduce.config import ReductionConfig

    beam_image, frame_header = frame_builders
    sam_z = [0.0, 0.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0]
    peaks = [10000.0, 10000.0, 4000.0, 2000.0]

    files = []
    for i, (z, th, pk) in enumerate(zip(sam_z, sam_th, peaks)):
        path = tmp_path / f"SIM_{i}.fits"
        fits_writer(path, beam_image(pk), frame_header(th, z))
        files.append(path)

    calls = {"n": 0}
    real_median_filter = st.median_filter

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_median_filter(*args, **kwargs)

    monkeypatch.setattr(st, "median_filter", counting)
    loader = SimplePXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
    loader.process(progress=False)
    # Exactly one median filter per frame — shared by finding and dezingering.
    assert calls["n"] == len(files)
