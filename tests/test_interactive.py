from dataclasses import replace

import pytest

from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader
from pxr_reduce.interactive import FrameView, analyze_frame


@pytest.fixture
def unprocessed_loader(tmp_path, fits_writer, frame_builders):
    """A loader that has NOT had process() run (viewer must work pre-process)."""
    beam_image, frame_header = frame_builders
    files = []
    for i, (z, th) in enumerate([(0, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4)]):
        p = tmp_path / f"IV_{i}.fits"
        fits_writer(p, beam_image(10000.0), frame_header(th, z))
        files.append(p)
    config = ReductionConfig(
        roi_height=9, roi_width=9, trim_x=5, trim_y=5,
        mask_threshold=100, dark_pix_offset=5,
    )
    return PXRLoader(files, config)


def test_analyze_frame_works_before_process(unprocessed_loader):
    assert not unprocessed_loader.data_processed
    view = analyze_frame(unprocessed_loader, 2)
    assert isinstance(view, FrameView)


def test_analyze_frame_shapes(unprocessed_loader):
    view = analyze_frame(unprocessed_loader, 2)
    # trimmed by 5 on each edge of a 61x61 image
    assert view.cleaned.shape == (51, 51)
    assert view.mask_preview.shape == view.cleaned.shape
    assert view.spot.shape == (9, 9)
    assert view.dark.shape == (9, 9)
    assert view.subtracted is not None
    assert view.subtracted.shape == (9, 9)


def test_analyze_frame_locates_beam_near_center(unprocessed_loader):
    view = analyze_frame(unprocessed_loader, 2)
    # synthetic beam is centered at (30, 30) in the raw frame
    assert abs(view.beam_raw[0] - 30) <= 2
    assert abs(view.beam_raw[1] - 30) <= 2


def test_analyze_frame_counts_and_meta(unprocessed_loader):
    view = analyze_frame(unprocessed_loader, 2)
    assert view.counts_spot > view.counts_dark
    assert view.net.value == pytest.approx(view.counts_spot - view.counts_dark)
    for key in ("energy", "sam_th", "q", "polarization"):
        assert key in view.meta


def test_analyze_frame_roi_boxes_within_bounds(unprocessed_loader):
    view = analyze_frame(unprocessed_loader, 2)
    h, w = view.raw.shape
    for x0, y0, x1, y1 in (view.roi_raw, view.dark_raw):
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_analyze_frame_respects_config_override(unprocessed_loader):
    cfg = replace(unprocessed_loader.config, roi_height=15, roi_width=15)
    view = analyze_frame(unprocessed_loader, 2, cfg)
    assert view.spot.shape == (15, 15)


def test_analyze_frame_mask_threshold_changes_mask(unprocessed_loader):
    # Small drift_distance so dilation does not flood the tiny synthetic frame.
    base = replace(unprocessed_loader.config, drift_distance=5)
    low = analyze_frame(unprocessed_loader, 2, replace(base, mask_threshold=10))
    high = analyze_frame(unprocessed_loader, 2, replace(base, mask_threshold=5000))
    # a higher threshold keeps fewer pixels in the mask
    assert low.mask_preview.sum() > high.mask_preview.sum()


def test_analyze_frame_unknown_index_raises(unprocessed_loader):
    with pytest.raises(KeyError):
        analyze_frame(unprocessed_loader, 999)


def test_viewer_constructs_without_plotly(unprocessed_loader):
    # __init__ must not require plotly/ipywidgets (only show() does).
    from pxr_reduce.interactive import InteractiveFrameViewer

    viewer = InteractiveFrameViewer(unprocessed_loader)
    assert viewer.indices == [0, 1, 2, 3, 4, 5]
    assert viewer._current == 0


def test_viewer_empty_selection_raises(unprocessed_loader):
    from pxr_reduce.interactive import InteractiveFrameViewer

    with pytest.raises(ValueError):
        InteractiveFrameViewer(unprocessed_loader, energy=999.0)
