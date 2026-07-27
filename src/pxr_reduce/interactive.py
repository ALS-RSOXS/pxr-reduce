"""Interactive Plotly-based frame viewer for inspecting and tuning reductions.

Two layers:

* :func:`analyze_frame` — a pure function that computes everything needed to view
  a single frame (cleaned image, per-frame mask preview, beam position, beam/dark
  ROIs, subtracted image, and scalar readouts) from an image + a
  :class:`~pxr_reduce.config.ReductionConfig`. It reuses :mod:`pxr_reduce.image_ops`
  and works **before** :meth:`~pxr_reduce.core.PXRLoader.process` has been run, so
  the viewer can navigate un-processed data. It has no Plotly dependency and is
  unit-testable headlessly.
* :class:`InteractiveFrameViewer` — a Plotly ``FigureWidget`` + ``ipywidgets`` UI
  built on top of :func:`analyze_frame`. Plotly/ipywidgets are imported lazily so
  they remain optional (install the ``notebook`` dependency group).

The mask shown is a **per-frame preview** (thresholding this frame alone); the
full reduction builds a single series mask from many frames. Likewise i0
normalization and stitch scaling need the whole stack and are not reproduced here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pxr_reduce import image_ops, tracking
from pxr_reduce.config import ReductionConfig
from pxr_reduce.uncertainty import Value, net_counts

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)

# Metadata columns shown in the readout panel, in display order, with labels.
_META_FIELDS: list[tuple[str, str, str]] = [
    ("sam_th", "Sample theta", "{:.4f} deg"),
    ("det_th", "CCD theta", "{:.4f} deg"),
    ("energy", "Energy", "{:.2f} eV"),
    ("polarization", "Polarization", "{:g}"),
    ("q", "q", "{:.5f} A^-1"),
    ("exposure", "Exposure", "{:.4f} s"),
    ("beam_current", "Beam current", "{:.1f} mA"),
    ("i0", "AI 3 Izero", "{:.4g}"),
    ("hos", "Higher-order suppressor", "{:g}"),
    ("slits_vert", "JJ vert aperture", "{:.4f} mm"),
    ("slits_horz", "JJ horz aperture", "{:.4f} mm"),
    ("scan", "Scan", "{:d}"),
]


@dataclass(frozen=True)
class FrameView:
    """Everything needed to display and describe a single frame.

    Coordinates in ``*_raw`` are in the original (untrimmed) image frame so they
    overlay directly on the raw image; ``beam`` is in trimmed-image coordinates.

    Args:
        fits_index: Frame index.
        raw: The original image (ADU).
        cleaned: Trimmed + (optionally) dezingered image.
        mask_preview: Per-frame integration mask (trimmed-image shape).
        beam: Beam ``(y, x)`` in trimmed-image coordinates.
        beam_raw: Beam ``(y, x)`` in original-image coordinates.
        roi_raw: Beam ROI ``(x0, y0, x1, y1)`` in original-image coordinates.
        dark_raw: Dark ROI ``(x0, y0, x1, y1)`` in original-image coordinates.
        spot: Beam ROI sub-image.
        dark: Dark ROI sub-image.
        subtracted: ``spot - dark`` (None if the ROIs have mismatched shapes).
        counts_spot: Summed beam ROI counts.
        counts_dark: Summed dark ROI counts.
        net: Background-subtracted counts and uncertainty.
        counts_ratio: ``counts_spot / counts_dark``.
        is_saturated: Whether the frame approaches detector saturation.
        meta: Scalar metadata for the frame.
    """

    fits_index: int
    raw: NDArray[np.floating]
    cleaned: NDArray[np.floating]
    mask_preview: NDArray[np.bool_]
    beam: tuple[int, int]
    beam_raw: tuple[int, int]
    roi_raw: tuple[int, int, int, int]
    dark_raw: tuple[int, int, int, int]
    spot: NDArray[np.floating]
    dark: NDArray[np.floating]
    subtracted: NDArray[np.floating] | None
    counts_spot: float
    counts_dark: float
    net: Value
    counts_ratio: float
    is_saturated: bool
    meta: dict[str, Any]


def _raw_rect(sl: tuple[slice, slice], trim_x: int, trim_y: int) -> tuple[int, int, int, int]:
    """Convert (row_slice, col_slice) in trimmed coords to a raw (x0,y0,x1,y1)."""
    sly, slx = sl
    return (
        slx.start + trim_y,
        sly.start + trim_x,
        slx.stop + trim_y,
        sly.stop + trim_x,
    )


def analyze_frame(
    loader: PXRLoader,
    fits_index: int,
    config: ReductionConfig | None = None,
) -> FrameView:
    """Compute the per-frame view (image crops, mask, beam, counts) for one frame.

    Mirrors the real pipeline for a single frame: the mask preview is built from
    the raw (trimmed) frame and the beam is located on the cleaned frame, exactly
    as :meth:`PXRLoader.process` does — but for this frame alone, so it works
    before processing and reflects the given ``config``.

    Args:
        loader: The loader providing the image and metadata.
        fits_index: Frame to analyze.
        config: Config to use; defaults to ``loader.config``.

    Returns:
        A populated :class:`FrameView`.
    """
    config = config or loader.config
    tx, ty = config.trim_x, config.trim_y

    raw = loader.get_image(fits_index)
    trimmed = image_ops.trim(raw, tx, ty)
    cleaned = image_ops.dezinger(trimmed, config)
    mask_preview = image_ops.build_series_mask([trimmed], config)
    # Robust centroid of the brightest blob within the mask (single-frame preview;
    # the full reduction tracks the beam across frames).
    beam = tracking.anchor_position(cleaned, mask_preview)

    roi_sl = image_ops.roi_slices(beam, config)
    dark_sl = image_ops.dark_roi_slices(beam, config)
    spot = cleaned[roi_sl]
    dark = cleaned[dark_sl]

    counts_spot = float(spot.sum())
    counts_dark = float(dark.sum())
    exposure = float(_meta_row(loader, fits_index).get("exposure", 1.0) or 1.0)
    net = net_counts(spot, dark, config.detector_spec(), exposure)
    # The subtracted display image needs matching ROI shapes.
    subtracted = spot - dark if (spot.shape == dark.shape and spot.size) else None

    ratio = counts_spot / counts_dark if counts_dark else float("inf")
    is_saturated = config.detector_spec().is_saturated(
        cleaned, config.saturate_threshold
    )

    return FrameView(
        fits_index=fits_index,
        raw=raw,
        cleaned=cleaned,
        mask_preview=mask_preview,
        beam=beam,
        beam_raw=(beam[0] + tx, beam[1] + ty),
        roi_raw=_raw_rect(roi_sl, tx, ty),
        dark_raw=_raw_rect(dark_sl, tx, ty),
        spot=spot,
        dark=dark,
        subtracted=subtracted,
        counts_spot=counts_spot,
        counts_dark=counts_dark,
        net=net,
        counts_ratio=ratio,
        is_saturated=is_saturated,
        meta=_meta_row(loader, fits_index),
    )


def _meta_row(loader: PXRLoader, fits_index: int) -> dict[str, Any]:
    """Return the metadata row for a frame as a plain dict."""
    row = loader.data.loc[loader.data["fits_index"] == fits_index]
    if row.empty:
        raise KeyError(f"fits_index {fits_index} not found in loader.")
    return row.iloc[0].to_dict()


# Config fields exposed as editable controls in the viewer (name, widget kind).
_CONFIG_CONTROLS: list[tuple[str, str]] = [
    ("mask_threshold", "int"),
    ("roi_height", "int"),
    ("roi_width", "int"),
    ("dark_pix_offset", "int"),
    ("darkside", "darkside"),
    ("drift_distance", "int"),
    ("trim_x", "int"),
    ("trim_y", "int"),
    ("filter_size", "int"),
    ("dezinger", "bool"),
]


class InteractiveFrameViewer:
    """Plotly ``FigureWidget`` viewer for inspecting and tuning frames.

    Layout: the original image on the left (with a red beam-ROI box and a blue
    dark-ROI box), the beam ROI (top) and dark ROI (bottom) in the middle, and
    the background-subtracted image on the right. A dropdown selects the frame; a
    panel shows the frame's metadata; config controls plus a "Reprocess frame"
    button re-run the per-frame analysis with edited parameters.

    Works before :meth:`~pxr_reduce.core.PXRLoader.process` has been run (it uses
    a per-frame mask preview). Requires the ``notebook`` dependency group
    (plotly, ipywidgets).

    Args:
        loader: The loader providing images and metadata.
        indices: Explicit frame indices to browse; if None, ``conditions`` select
            them (all frames if neither is given).
        config: Starting config; defaults to ``loader.config``.
        **conditions: Metadata filters forwarded to :func:`select_indices`.

    Raises:
        ValueError: If the selection is empty.
    """

    # Trace indices within the figure (see :meth:`_build_figure`).
    _IMAGE, _BEAM, _SPOT, _DARK, _SUB = 0, 1, 2, 3, 4

    def __init__(
        self,
        loader: PXRLoader,
        indices: list[int] | None = None,
        config: ReductionConfig | None = None,
        **conditions: Any,
    ) -> None:
        from pxr_reduce.viewer import select_indices

        self.loader = loader
        self.indices = (
            indices if indices is not None else select_indices(loader, **conditions)
        )
        if not self.indices:
            raise ValueError("No frames match the viewer selection.")
        self._working_config = replace(config or loader.config)
        self._current = self.indices[0]
        self._fig = None
        self._controls: dict[str, Any] = {}
        self._meta_html = None
        self._show_mask = None

    # -- Rendering helpers ----------------------------------------------------

    def _to_rgb(
        self, image: NDArray[np.floating], mask_raw: NDArray[np.bool_] | None
    ) -> NDArray[np.uint8]:
        """Log-normalize an image to an RGB array, optionally tinting the mask."""
        from matplotlib import colormaps, colors

        positive = image[image > 0]
        vmin = float(positive.min()) if positive.size else 1.0
        vmax = float(image.max()) if image.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
        rgba = colormaps["terrain"](norm(np.clip(image, 1e-6, None)))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        if mask_raw is not None:
            tint = np.array([255, 0, 0], dtype=np.float64)
            rgb[mask_raw] = (0.6 * rgb[mask_raw] + 0.4 * tint).astype(np.uint8)
        return rgb

    def _format_meta(self, view: FrameView) -> str:
        """Build an HTML table of the frame's metadata and computed readouts."""
        rows = [f"<b>Frame #{view.fits_index}</b>"]
        rows.append("<table style='font-family:monospace;font-size:12px'>")
        for key, label, spec in _META_FIELDS:
            val = view.meta.get(key)
            if val is None:
                continue
            try:
                text = spec.format(val)
            except (ValueError, TypeError):
                text = str(val)
            rows.append(f"<tr><td>{label}</td><td>{text}</td></tr>")
        # Per-frame-preview computed values.
        rows.append("<tr><td colspan=2><i>per-frame preview</i></td></tr>")
        computed = {
            "Beam (y,x)": f"{view.beam_raw}",
            "Spot counts": f"{view.counts_spot:.1f}",
            "Dark counts": f"{view.counts_dark:.1f}",
            "Net": f"{view.net.value:.4g}",
            "SNR (spot/dark)": f"{view.counts_ratio:.3f}",
            "Saturated": f"{view.is_saturated}",
        }
        for label, text in computed.items():
            rows.append(f"<tr><td>{label}</td><td>{text}</td></tr>")
        rows.append("</table>")
        return "\n".join(rows)

    # -- Figure construction / update -----------------------------------------

    def _build_figure(self):  # pragma: no cover - requires plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=3,
                specs=[
                    [{"rowspan": 2}, {}, {"rowspan": 2}],
                    [None, {}, None],
                ],
                column_widths=[0.5, 0.2, 0.3],
                row_heights=[0.5, 0.5],
                subplot_titles=("Original", "Beam ROI", "Subtracted", "Dark"),
                horizontal_spacing=0.06,
                vertical_spacing=0.1,
            )
        )
        blank = np.zeros((1, 1), dtype=np.uint8)
        fig.add_trace(go.Image(z=np.zeros((1, 1, 3), dtype=np.uint8)), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers",
                       marker=dict(color="red", symbol="cross", size=10),
                       showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(go.Heatmap(z=blank, colorscale="Viridis", showscale=False),
                      row=1, col=2)
        fig.add_trace(go.Heatmap(z=blank, colorscale="Viridis", showscale=False),
                      row=2, col=2)
        fig.add_trace(
            go.Heatmap(z=blank, colorscale="RdBu", zmid=0, showscale=True),
            row=1, col=3,
        )
        # Row 0 at the top for every panel (matches image orientation).
        for r, c in ((1, 1), (1, 2), (2, 2), (1, 3)):
            fig.update_yaxes(autorange="reversed", row=r, col=c)
            fig.update_xaxes(showticklabels=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, row=r, col=c)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    def _render(self) -> None:  # pragma: no cover - requires plotly
        view = analyze_frame(self.loader, self._current, self._working_config)
        tx, ty = self._working_config.trim_x, self._working_config.trim_y

        mask_raw = None
        if self._show_mask.value:
            mask_raw = np.zeros(view.raw.shape, dtype=bool)
            mh, mw = view.mask_preview.shape
            mask_raw[tx : tx + mh, ty : ty + mw] = view.mask_preview
        rgb = self._to_rgb(view.raw, mask_raw)
        sub = view.subtracted if view.subtracted is not None else np.zeros_like(view.spot)
        sub_max = float(np.abs(sub).max()) or 1.0

        fig = self._fig
        with fig.batch_update():
            fig.data[self._IMAGE].z = rgb
            fig.data[self._BEAM].x = [view.beam_raw[1]]
            fig.data[self._BEAM].y = [view.beam_raw[0]]
            fig.data[self._SPOT].z = view.spot
            fig.data[self._DARK].z = view.dark
            fig.data[self._SUB].z = sub
            fig.data[self._SUB].zmin = -sub_max
            fig.data[self._SUB].zmax = sub_max
            rx0, ry0, rx1, ry1 = view.roi_raw
            dx0, dy0, dx1, dy1 = view.dark_raw
            fig.layout.shapes = (
                dict(type="rect", xref="x", yref="y", x0=rx0, y0=ry0, x1=rx1, y1=ry1,
                     line=dict(color="red", width=2)),
                dict(type="rect", xref="x", yref="y", x0=dx0, y0=dy0, x1=dx1, y1=dy1,
                     line=dict(color="blue", width=2)),
            )
        self._meta_html.value = self._format_meta(view)

    # -- Public entry point ---------------------------------------------------

    def show(self):  # pragma: no cover - requires plotly/ipywidgets
        """Build and return the interactive widget (display it in a notebook)."""
        import ipywidgets as widgets

        self._fig = self._build_figure()

        # Frame dropdown, labeled with index + angle + energy + polarization.
        options = []
        for idx in self.indices:
            m = _meta_row(self.loader, idx)
            options.append(
                (
                    f"#{idx}  θ={m.get('sam_th', float('nan')):.3f}°  "
                    f"E={m.get('energy', float('nan')):g}eV  "
                    f"pol={m.get('polarization', float('nan')):g}",
                    idx,
                )
            )
        dropdown = widgets.Dropdown(options=options, value=self._current,
                                    description="Frame:")

        def _on_frame(change):
            self._current = change["new"]
            self._render()

        dropdown.observe(_on_frame, names="value")

        # Config controls.
        self._controls = {}
        for name, kind in _CONFIG_CONTROLS:
            current = getattr(self._working_config, name)
            if kind == "int":
                w = widgets.IntText(value=int(current), description=name)
            elif kind == "bool":
                w = widgets.Checkbox(value=bool(current), description=name)
            elif kind == "darkside":
                w = widgets.Dropdown(options=["LHS", "RHS"], value=current,
                                     description=name)
            self._controls[name] = w

        self._show_mask = widgets.Checkbox(value=True, description="show mask overlay")
        reprocess = widgets.Button(description="Reprocess frame",
                                   button_style="primary")

        def _on_reprocess(_btn):
            updates = {name: w.value for name, w in self._controls.items()}
            try:
                self._working_config = replace(self._working_config, **updates)
            except (ValueError, TypeError) as exc:
                self._meta_html.value = f"<b style='color:red'>Invalid config: {exc}</b>"
                return
            self._render()

        reprocess.on_clicked(_on_reprocess)
        self._show_mask.observe(lambda _c: self._render(), names="value")

        self._meta_html = widgets.HTML()

        controls = widgets.VBox(
            [dropdown, self._show_mask, *self._controls.values(), reprocess]
        )
        self._render()
        return widgets.VBox([widgets.HBox([self._fig, self._meta_html]), controls])
