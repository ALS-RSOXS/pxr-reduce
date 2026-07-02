"""Metadata-driven image viewer for inspecting and comparing frames.

Query frames by metadata (e.g. a sample-theta range at a given energy) and cycle
through them, overlaying the beam position and ROI/dark boxes and showing scalar
readouts (raw counts, reduced intensity, SNR, saturation).

Two entry points:

* :func:`frame_figure` renders one frame to a headless :class:`~matplotlib.figure.Figure`
  (for tests, scripts, or saving).
* :class:`FrameBrowser` is an interactive matplotlib widget for cycling through a
  selection (for notebook-free live inspection).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pxr_reduce import image_ops

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)


def select_indices(loader: PXRLoader, **conditions: Any) -> list[int]:
    """Return sorted frame indices matching metadata conditions.

    Args:
        loader: The loader to query.
        **conditions: Column filters passed to :meth:`PXRLoader.query` (scalar
            for equality, 2-tuple for an inclusive range).

    Returns:
        Sorted list of matching ``fits_index`` values.
    """
    subset = loader.query(**conditions) if conditions else loader.data
    return sorted(int(i) for i in subset["fits_index"])


def _frame_scalars(loader: PXRLoader, fits_index: int) -> dict[str, Any]:
    """Collect the scalar readouts for a frame from the metadata table."""
    row = loader.data.loc[loader.data["fits_index"] == fits_index].iloc[0]
    keys = [
        "sam_th",
        "det_th",
        "energy",
        "polarization",
        "q",
        "counts_spot",
        "counts_dark",
        "counts_refl",
        "counts_ratio",
        "is_saturated",
        "beam_spot",
    ]
    return {k: row[k] for k in keys if k in row}


def _format_scalars(fits_index: int, info: dict[str, Any]) -> str:
    """Format scalar readouts as a compact multi-line string."""
    lines = [f"frame {fits_index}"]
    fmt = {
        "sam_th": ("theta", "{:.4f} deg"),
        "energy": ("E", "{:.2f} eV"),
        "polarization": ("pol", "{:g}"),
        "q": ("q", "{:.5f} A^-1"),
        "counts_spot": ("spot", "{:.1f}"),
        "counts_dark": ("dark", "{:.1f}"),
        "counts_refl": ("reduced", "{:.4g}"),
        "counts_ratio": ("SNR", "{:.3f}"),
        "is_saturated": ("saturated", "{}"),
    }
    for key, (label, spec) in fmt.items():
        if key in info and info[key] is not None:
            lines.append(f"{label}: {spec.format(info[key])}")
    return "\n".join(lines)


def _draw_image(ax: Axes, loader: PXRLoader, fits_index: int) -> dict[str, Any]:
    """Draw a frame's cleaned image with mask/beam/ROI overlays onto ``ax``.

    Args:
        ax: Target axes.
        loader: The loader providing the image and metadata.
        fits_index: Frame to draw.

    Returns:
        The scalar readouts for the frame.
    """
    from matplotlib.colors import LogNorm

    image = loader.get_clean_image(fits_index)
    positive = image[image > 0]
    vmin = positive.min() if positive.size else 1.0
    ax.imshow(image, norm=LogNorm(vmin=max(vmin, 1e-6)), cmap="terrain")

    if loader.mask is not None and loader.mask.shape == image.shape:
        ax.imshow(np.ma.masked_where(loader.mask, loader.mask), cmap="Greys_r", alpha=0.3)

    info = _frame_scalars(loader, fits_index)
    beam_spot = info.get("beam_spot")
    if beam_spot is not None:
        y, x = beam_spot
        ax.plot(x, y, "r+", ms=10)
        for slicer, color in (
            (image_ops.roi_slices, "red"),
            (image_ops.dark_roi_slices, "cyan"),
        ):
            sly, slx = slicer(beam_spot, loader.config)
            ax.add_patch(
                Rectangle(
                    (slx.start, sly.start),
                    slx.stop - slx.start,
                    sly.stop - sly.start,
                    fill=False,
                    edgecolor=color,
                    lw=1,
                )
            )
    ax.set_xticks([])
    ax.set_yticks([])
    return info


def frame_figure(loader: PXRLoader, fits_index: int) -> Figure:
    """Render a single frame with overlays and scalar readouts.

    Args:
        loader: The loader providing the image and metadata.
        fits_index: Frame to render.

    Returns:
        A headless :class:`~matplotlib.figure.Figure`.
    """
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    image_ax, text_ax = fig.subplots(1, 2, width_ratios=[3, 1])
    info = _draw_image(image_ax, loader, fits_index)
    text_ax.axis("off")
    text_ax.text(
        0.0, 1.0, _format_scalars(fits_index, info), va="top", ha="left", fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    return fig


class FrameBrowser:
    """Interactive browser to cycle through a metadata-selected set of frames.

    Uses matplotlib widgets (slider + prev/next buttons) for notebook-free live
    inspection. Construct with a query, then call :meth:`show`.

    Args:
        loader: The loader providing images and metadata.
        indices: Explicit frame indices to browse; if None, ``conditions`` are
            used to select them (all frames if neither is given).
        **conditions: Metadata filters forwarded to :func:`select_indices`.

    Raises:
        ValueError: If the selection is empty.
    """

    def __init__(
        self,
        loader: PXRLoader,
        indices: list[int] | None = None,
        **conditions: Any,
    ) -> None:
        self.loader = loader
        self.indices = indices if indices is not None else select_indices(
            loader, **conditions
        )
        if not self.indices:
            raise ValueError("No frames match the viewer selection.")
        self._pos = 0
        self._fig = None
        self._ax = None

    def _render(self) -> None:
        self._ax.clear()
        info = _draw_image(self._ax, self.loader, self.indices[self._pos])
        self._ax.set_title(
            _format_scalars(self.indices[self._pos], info), fontsize=8, loc="left"
        )
        self._fig.canvas.draw_idle()

    def _step(self, delta: int) -> None:
        self._pos = (self._pos + delta) % len(self.indices)
        self._render()

    def show(self) -> None:  # pragma: no cover - interactive
        """Open the interactive browser window."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._fig.subplots_adjust(bottom=0.15)
        prev_ax = self._fig.add_axes((0.3, 0.03, 0.15, 0.06))
        next_ax = self._fig.add_axes((0.55, 0.03, 0.15, 0.06))
        self._prev_btn = Button(prev_ax, "< Prev")
        self._next_btn = Button(next_ax, "Next >")
        self._prev_btn.on_clicked(lambda _event: self._step(-1))
        self._next_btn.on_clicked(lambda _event: self._step(1))
        self._render()
        plt.show()
