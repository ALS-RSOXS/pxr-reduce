import pytest
from matplotlib.figure import Figure

from pxr_reduce.viewer import FrameBrowser, frame_figure, select_indices


@pytest.fixture
def loader(processed_loader_factory):
    return processed_loader_factory()


def test_select_indices_all(loader):
    assert select_indices(loader) == [0, 1, 2, 3, 4, 5]


def test_select_indices_by_range(loader):
    idx = select_indices(loader, sam_th=(1.0, 3.0))
    assert idx == [2, 3, 4]


def test_frame_figure_returns_figure_with_overlays(loader):
    fig = frame_figure(loader, 3)
    assert isinstance(fig, Figure)
    # image axes + text axes
    assert len(fig.axes) == 2
    # beam spot marker and ROI rectangles drawn on the image axes
    image_ax = fig.axes[0]
    assert len(image_ax.patches) >= 2  # beam ROI + dark ROI rectangles


def test_frame_figure_on_unprocessed_loader(synthetic_scan_folder):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader

    folder = synthetic_scan_folder()
    loader = PXRLoader(sorted(folder.glob("*.fits")), ReductionConfig())
    # not processed: no beam_spot column, should still render the image
    fig = frame_figure(loader, 0)
    assert isinstance(fig, Figure)


def test_browser_construction_and_stepping(loader):
    browser = FrameBrowser(loader, sam_th=(1.0, 3.0))
    assert browser.indices == [2, 3, 4]
    assert browser._pos == 0
    # stepping wraps around without a live window
    browser._pos = (browser._pos - 1) % len(browser.indices)
    assert browser._pos == 2


def test_browser_empty_selection_raises(loader):
    with pytest.raises(ValueError):
        FrameBrowser(loader, energy=9999.0)
