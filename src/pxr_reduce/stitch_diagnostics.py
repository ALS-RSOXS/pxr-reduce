"""Per-boundary stitch diagnostics: why each stitch scaled the way it did.

Written into a ``stitch/`` subfolder of the ``--diagnostics`` output, with one
subfolder per scan (stitching is a per-scan operation):

.. code-block:: text

    <sample>_diagnostics/stitch/
        stitch_summary.md               per-boundary report, embeds the figures
        dropped_points.md               every rejected overlap candidate
        scan_00/
            boundary_01.png             the stitch itself
            saturated/
                frame_00041_roi.png     ROI of a saturation-dropped point

The per-boundary figure has two panels: R vs angle, showing every overlap candidate
with its fate (used, saturated, below the spot/dark cutoff, direct-beam, partner
dropped) and the post-change segment before and after scaling so you can see whether
the two segments overlay; and the fit itself — post-change R against pre-change R for
each matched angle, with the fitted through-origin line.

Saturated-frame ROI images are written only for frames that actually cost a stitch
point; every other saturated frame is listed in the summary instead, since plotting
them means re-reading frames from disk.

Figures use the headless Agg backend (no pyplot state machine), like
:mod:`pxr_reduce.diagnostics`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from pxr_reduce import image_ops, reduction

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)

# Runaway guard on saturated-ROI images; each one re-reads a frame from disk.
_MAX_SATURATED_IMAGES = 60

# Marker style per rejection reason, matched by prefix.
_REASON_STYLE: dict[str, tuple[str, str, str]] = {
    "saturated": ("X", "red", "saturated"),
    "counts_ratio": ("s", "darkorange", "below spot/dark cutoff"),
    "direct-beam": ("^", "purple", "direct-beam (i0)"),
    "partner dropped": ("v", "teal", "partner dropped"),
}


def _reason_style(reason: str) -> tuple[str, str, str]:
    """Return the ``(marker, colour, legend label)`` for a rejection reason."""
    for prefix, style in _REASON_STYLE.items():
        if reason.startswith(prefix):
            return style
    return ("d", "0.4", "excluded")


def boundary_figure(
    group: pd.DataFrame,
    points: pd.DataFrame,
    boundary: pd.Series,
    *,
    sample: str,
) -> Figure:
    """Build the stitch figure for one boundary.

    Args:
        group: Annotated rows for the whole scan (from
            :func:`pxr_reduce.reduction.annotate`), positionally indexed.
        points: Overlap candidates for this boundary (from
            :func:`pxr_reduce.reduction.overlap_report`).
        boundary: The boundary's row from
            :func:`pxr_reduce.reduction.diagnose_stitches`.
        sample: Sample name for the title.

    Returns:
        The rendered :class:`~matplotlib.figure.Figure`.
    """
    b = int(boundary["boundary_index"])
    scale = float(boundary["scale"])
    prev_scale = float(group["scale"].iloc[b - 1]) if b > 0 else 1.0
    ratio = scale / prev_scale if prev_scale else float("nan")

    fig = Figure(figsize=(9, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    # No shared x-axis: the top panel is R vs angle, the bottom is post vs pre R.
    top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1.25]})

    # --- Top: the two segments, and the post segment after scaling -------------
    segment = (group["mark"] == 1).cumsum().to_numpy()
    seg_here = segment[b]
    theta = group["sam_th"].to_numpy(dtype=float)
    r = group["R"].to_numpy(dtype=float)
    # Direct-beam frames are not reflectivity, and saturated frames are outliers by
    # construction: exclude both from the connecting lines so the curves read as
    # R(theta). Saturated frames are still drawn individually, marked as excluded.
    is_refl = (
        group["i0_mask"].to_numpy() < 1 if "i0_mask" in group.columns
        else np.ones(len(group), dtype=bool)
    )
    clean = is_refl & ~group["is_saturated"].to_numpy(dtype=bool)

    for seg, colour, label in (
        (seg_here - 1, "tab:blue", "pre-change segment"),
        (seg_here, "tab:red", "post-change segment (raw)"),
    ):
        sel = clean & (segment == seg)
        order = np.argsort(theta[sel])
        top.plot(theta[sel][order], r[sel][order], "o-", ms=4, lw=0.8,
                 color=colour, label=label, zorder=2)

    post = clean & (segment == seg_here)
    if np.isfinite(ratio) and ratio > 0 and post.any():
        order = np.argsort(theta[post])
        top.plot(theta[post][order], (r[post] / ratio)[order], "o--", ms=4, lw=0.8,
                 color="tab:green", mfc="none",
                 label=f"post / {ratio:.4g} (should overlay pre)", zorder=3)

    top.axvline(theta[b], ls="--", color="0.5", lw=1, zorder=1)

    # Overlap candidates: used points ringed, rejected points marked by reason.
    used = points[points["used"]]
    if len(used):
        n_angles = int(boundary["num_stitch_points"])
        top.scatter(used["sam_th"], used["R"], s=130, facecolors="none",
                    edgecolors="black", lw=1.4, zorder=4,
                    label=f"used in fit ({len(used)} frames, {n_angles} angle(s))")
    seen: set[str] = set()
    for _, row in points[~points["used"]].iterrows():
        marker, colour, label = _reason_style(str(row["reason"]))
        top.scatter(row["sam_th"], row["R"], s=70, marker=marker, color=colour,
                    zorder=5, label=None if label in seen else label)
        seen.add(label)
        top.annotate(
            f"#{int(row['fits_index'])}", xy=(row["sam_th"], row["R"]),
            textcoords="offset points", xytext=(4, -9), fontsize=6, color=colour,
        )

    top.set_yscale("log")
    top.set_xlabel("sam_th [deg]")
    top.set_ylabel("R (I0-normalized, pre-scaling)")
    top.set_title(
        f"{sample}   scan {boundary['scan']}   boundary at "
        f"sam_th={boundary['sam_th']:.4f} deg   E={boundary['energy']:g} eV",
        fontsize=10,
    )
    top.legend(fontsize=7, loc="best")
    top.text(
        0.01, 0.02, _fit_caption(boundary),
        transform=top.transAxes, fontsize=7, va="bottom", family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7", "pad": 4},
    )

    # --- Bottom: the fit itself, post-change R against pre-change R ------------
    _fit_panel(bottom, points, ratio, boundary)
    return fig


def _fit_caption(boundary: pd.Series) -> str:
    """Return the multi-line fit/quality caption for a boundary figure."""
    lines = [
        f"trigger    : {boundary['trigger']}",
        f"conditions : {boundary['conditions_changed'] or '(none)'}",
        f"overlap    : {int(boundary['num_stitch_points'])} angle(s)",
        f"scale      : {boundary['scale']:.6g} +/- {boundary['scale_err']:.3g}",
    ]
    rms = boundary["overlap_rms_rel"]
    lines.append(
        f"overlap rms: {rms:.2%}" if np.isfinite(rms)
        else "overlap rms: n/a (needs 2+ angles)"
    )
    expected = boundary["expected_scale"]
    lines.append(
        f"expected   : {expected:.6g}" if np.isfinite(expected)
        else "expected   : not predictable (flux changed)"
    )
    if boundary["failed"]:
        lines.append(f"FAILED     : {boundary['fail_reason']}")
    elif boundary["suspect"]:
        lines.append(f"SUSPECT    : {boundary['quality_note']}")
    else:
        lines.append("quality    : ok")
    if not boundary["scale_established"]:
        lines.append("NOTE       : absolute scale unestablished (earlier failure)")
    return "\n".join(lines)


def _fit_panel(
    ax: Any, points: pd.DataFrame, ratio: float, boundary: pd.Series
) -> None:
    """Draw the fit itself: post-change intensity against pre-change intensity.

    This is the fit as ``curve_fit`` sees it — one point per matched angle, and the
    through-origin line ``post = scale * pre`` fitted to them. A good stitch has the
    points sitting on the line; a point off the line is an angle where the two
    segments disagree, and with only one or two points there is nothing constraining
    the line at all.

    Args:
        ax: Axes to draw on.
        points: Overlap candidates for this boundary.
        ratio: The scale factor fitted at this boundary.
        boundary: The boundary's diagnostic row.
    """
    ax.set_xlabel("pre-change R (I0-normalized)")
    ax.set_ylabel("post-change R")

    used = points[points["used"]]
    pre = used[used["side"] == "pre"].groupby("sam_th")["R"].mean()
    post = used[used["side"] == "post"].groupby("sam_th")["R"].mean()
    common = sorted(set(pre.index) & set(post.index))
    if not common:
        ax.text(0.5, 0.5, "no fitted overlap points", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
        return

    x = np.array([pre.loc[a] for a in common], dtype=float)
    y = np.array([post.loc[a] for a in common], dtype=float)
    span = np.array([0.0, float(x.max()) * 1.1])

    if np.isfinite(ratio) and ratio > 0:
        ax.plot(span, ratio * span, "-", color="tab:green", lw=1.3,
                label=f"fit: post = {ratio:.4g} × pre")
    expected = boundary["expected_scale"]
    if np.isfinite(expected):
        ax.plot(span, expected * span, "--", color="0.55", lw=1.1,
                label=f"expected: × {expected:.4g}")

    ax.plot(x, y, "o", color="black", ms=6, zorder=3,
            label=f"overlap angles ({len(common)})")
    for angle, xi, yi in zip(common, x, y):
        ax.annotate(f"{angle:.4g}°", xy=(xi, yi), textcoords="offset points",
                    xytext=(6, -3), fontsize=6.5, color="0.25")

    # Include the origin: the fit is constrained through it, so a reader should be
    # able to see whether that constraint is consistent with the data.
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="best")


def saturated_roi_figure(
    loader: PXRLoader, fits_index: int, row: pd.Series
) -> Figure:
    """Build the beam-ROI image for one saturated frame.

    Shows only the integrated beam ROI — the region whose saturation actually
    corrupts ``counts_spot`` — with the beam centre and every saturated pixel marked.

    Args:
        loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
        fits_index: Index of the frame to render.
        row: That frame's row from the annotated table.

    Returns:
        The rendered :class:`~matplotlib.figure.Figure`.
    """
    config = loader.config
    detector = config.detector_spec()
    image = loader.get_clean_image(fits_index)
    beam = tuple(int(v) for v in row["beam_spot"])
    rows, cols = image_ops.roi_slices(beam, config)
    roi = image[rows, cols]
    saturated = detector.saturated_mask(roi, config.saturate_threshold)

    fig = Figure(figsize=(6.0, 5.6), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if roi.size == 0:
        ax.text(0.5, 0.5, "beam ROI is clipped away at the frame edge",
                ha="center", va="center", transform=ax.transAxes, color="red")
    else:
        floor = max(float(roi[roi > 0].min()) if (roi > 0).any() else 1.0, 1.0)
        im = ax.imshow(
            np.clip(roi, floor, None), origin="upper", cmap="viridis",
            norm=LogNorm(vmin=floor, vmax=max(float(roi.max()), floor * 10)),
        )
        fig.colorbar(im, ax=ax, label="counts [ADU]", fraction=0.046)
        ys, xs = np.nonzero(saturated)
        if len(ys):
            ax.scatter(xs, ys, s=14, marker="s", facecolors="none",
                       edgecolors="red", lw=0.6,
                       label=f"saturated ({len(ys)} px)")
        # Beam centre in ROI coordinates.
        ax.plot(beam[1] - cols.start, beam[0] - rows.start, "+", color="white",
                ms=14, mew=2, label="beam centre")
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("x within ROI [pix]")
    ax.set_ylabel("y within ROI [pix]")
    ax.set_title(
        f"frame {fits_index}  {Path(loader.path_for(fits_index)).name}\n"
        f"sam_th={row['sam_th']:.4f} deg  E={row['energy']:g} eV  "
        f"exposure={row['exposure']:g} s\n"
        f"saturated: {int(row.get('n_sat_roi', 0))} px in ROI, "
        f"{int(row.get('n_sat_dark', 0))} px in dark ROI "
        f"(threshold {detector.saturation_adu - config.saturate_threshold:g} ADU)",
        fontsize=8,
    )
    return fig


def _by_scan(boundaries: pd.DataFrame) -> list[tuple[Any, pd.DataFrame]]:
    """Group boundaries by scan, tolerating a sample with no stitches at all.

    :func:`~pxr_reduce.reduction.diagnose_stitches` returns a column-less frame when
    no boundary was detected, so a bare ``groupby("scan")`` would raise.
    """
    if not len(boundaries):
        return []
    return list(boundaries.groupby("scan", sort=True))


def _scan_dir_name(scan_id: Any) -> str:
    """Return the per-scan subfolder name for a scan label."""
    try:
        return f"scan_{int(scan_id):02d}"
    except (TypeError, ValueError):
        return f"scan_{scan_id}"


def _boundary_plot_name(ordinal: int) -> str:
    """Return the per-boundary figure filename, keyed on the stitch number."""
    return f"boundary_{ordinal:02d}.png"


def _md_cell(value: Any) -> str:
    """Render a value as a Markdown table cell, escaping the column separator."""
    return str(value).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Build a Markdown table.

    Hand-rolled rather than via :meth:`pandas.DataFrame.to_markdown`, which needs the
    optional ``tabulate`` package — not worth a runtime dependency for formatting.

    Args:
        headers: Column headings.
        rows: Row values, one list per row.

    Returns:
        The table lines; a single italic note if there are no rows.
    """
    if not rows:
        return ["_none_"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += [
        "| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows
    ]
    return lines


def summary_lines(
    loader: PXRLoader,
    boundaries: pd.DataFrame,
    points: pd.DataFrame,
    annotated: pd.DataFrame,
    plot_names: dict[Any, str],
    image_names: dict[int, str],
) -> list[str]:
    """Build the stitch summary report.

    Every dropped overlap point is named with its source FITS file and, where one was
    written, its ROI image — so a questionable stitch can be traced to the frames
    responsible. Saturated frames that did not cost a stitch point are listed too,
    without images.

    Args:
        loader: The processed loader (for file paths and config).
        boundaries: :func:`~pxr_reduce.reduction.diagnose_stitches` output.
        points: :func:`~pxr_reduce.reduction.overlap_report` output.
        annotated: Per-frame annotated table.
        plot_names: Boundary row label -> scan-relative plot filename.
        image_names: ``fits_index`` -> scan-relative ROI image filename.

    Returns:
        The report lines (without trailing newlines).
    """
    config = loader.config
    detector = config.detector_spec()
    counts = reduction.summarize_stitches(boundaries)
    cut = detector.saturation_adu - config.saturate_threshold
    saturated = annotated[annotated["is_saturated"].astype(bool)]

    lines = [
        f"# Stitch diagnostics - {loader.name}",
        "",
        f"- **Source:** `{loader.path}`",
        f"- **Frames:** {len(annotated)}",
        f"- **Scans:** {annotated['scan'].nunique()}",
        f"- **Boundaries:** {counts['total']} "
        f"({counts['ok']} ok, {counts['suspect']} suspect, {counts['failed']} failed)",
        f"- **Saturated frames:** {len(saturated)}",
        "",
        "> A frame counts as saturated when any pixel of the **integrated beam ROI** "
        f"exceeds {cut:g} ADU",
        f"> (detector saturation {detector.saturation_adu:g} minus "
        f"`saturate_threshold` {config.saturate_threshold:g}). Only the beam ROI is",
        "> tested: saturation elsewhere on the detector does not affect the measured "
        "counts. Saturated",
        "> frames are excluded from stitch overlaps and from the reduced dataset.",
    ]

    if not len(boundaries):
        lines += ["", "No stitch boundaries were detected in this sample."]

    for scan_id, scan_boundaries in _by_scan(boundaries):
        scan_frames = annotated[annotated["scan"] == scan_id]
        energies = ", ".join(f"{e:g}" for e in sorted(scan_frames["energy"].unique()))
        lines += [
            "",
            f"## {_scan_dir_name(scan_id)}",
            "",
            f"- **Frames:** {len(scan_frames)} "
            f"(fits_index {int(scan_frames['fits_index'].min())}-"
            f"{int(scan_frames['fits_index'].max())})",
            f"- **Energies [eV]:** {energies}",
            f"- **Boundaries:** {len(scan_boundaries)}",
        ]
        for ordinal, (pos, boundary) in enumerate(scan_boundaries.iterrows(), start=1):
            lines += _boundary_lines(
                loader, boundary, points, plot_names.get(pos, ""), image_names, ordinal
            )

    lines += _saturation_lines(loader, annotated, points, image_names)
    return lines


def _boundary_lines(
    loader: PXRLoader,
    boundary: pd.Series,
    points: pd.DataFrame,
    plot_name: str,
    image_names: dict[int, str],
    ordinal: int,
) -> list[str]:
    """Build the report block for one boundary, including every dropped point."""
    mine = points[
        (points["scan"] == boundary["scan"])
        & (points["boundary_index"] == boundary["boundary_index"])
    ]
    used = mine[mine["used"]]
    dropped = mine[~mine["used"]]

    rms = boundary["overlap_rms_rel"]
    expected = boundary["expected_scale"]
    if boundary["failed"]:
        verdict = f"**FAILED** - {boundary['fail_reason']}"
    elif boundary["suspect"]:
        verdict = f"**SUSPECT** - {boundary['quality_note']}"
    else:
        verdict = "ok"

    fields = [
        ["fits_index", int(boundary["fits_index"])],
        ["sam_th [deg]", f"{boundary['sam_th']:.4f}"],
        ["energy [eV]", f"{boundary['energy']:g}"],
        ["polarization", f"{boundary['polarization']:g}"],
        ["trigger", f"`{boundary['trigger']}`"],
        ["conditions changed", boundary["conditions_changed"] or "_none_"],
        [
            "overlap",
            f"{int(boundary['num_stitch_points'])} angle(s) fitted from "
            f"{len(used)} frame(s); {len(dropped)} candidate frame(s) dropped",
        ],
        [
            "scale",
            f"{boundary['scale']:.6g} +/- {boundary['scale_err']:.3g}",
        ],
        [
            "overlap rms",
            f"{rms:.2%}" if np.isfinite(rms) else "n/a (needs 2+ overlap angles)",
        ],
        [
            "expected scale",
            f"{expected:.6g}" if np.isfinite(expected)
            else "not predictable (incident flux changed)",
        ],
        ["verdict", verdict],
    ]
    if not boundary["scale_established"]:
        fields.append(
            [
                "note",
                "absolute scale unestablished here (an earlier boundary in this "
                "scan failed)",
            ]
        )

    lines = [
        "",
        f"### Boundary {ordinal:02d} - sam_th = {boundary['sam_th']:.4f} deg",
        "",
    ]
    lines += _markdown_table(["field", "value"], fields)
    if plot_name:
        lines += ["", f"![Boundary {ordinal:02d}]({plot_name})"]

    lines += ["", "**Dropped overlap points**", ""]
    rows: list[list[Any]] = []
    for _, row in dropped.sort_values(["sam_th", "fits_index"]).iterrows():
        idx = int(row["fits_index"])
        detail = str(row["reason"])
        if detail.startswith("saturated"):
            detail = f"saturated ({int(row['n_sat_roi'])} px in beam ROI)"
        image = f"[ROI]({image_names[idx]})" if idx in image_names else ""
        rows.append(
            [
                row["side"],
                f"{row['sam_th']:.4f}",
                idx,
                detail,
                f"`{Path(loader.path_for(idx)).name}`",
                image,
            ]
        )
    lines += _markdown_table(
        ["side", "sam_th", "fits_index", "reason", "file", "image"], rows
    )
    return lines


def _saturation_lines(
    loader: PXRLoader,
    annotated: pd.DataFrame,
    points: pd.DataFrame,
    image_names: dict[int, str],
) -> list[str]:
    """Build the saturated-frame section, including frames that cost no stitch point."""
    saturated = annotated[annotated["is_saturated"].astype(bool)]
    lines = ["", "## Saturated frames", ""]
    if not len(saturated):
        lines.append("_None: no frame has a saturated pixel in its beam ROI._")
        return lines

    costly = _saturation_dropped_indices(points)
    lines += [
        f"{len(saturated)} frame(s) have saturated pixels in the beam ROI; "
        f"{len(costly)} of them cost a stitch",
        "overlap point and have an ROI image written. The rest are listed for "
        "reference only. All of",
        "them are dropped from the reduced dataset.",
        "",
    ]
    rows: list[list[Any]] = []
    for _, row in saturated.sort_values("fits_index").iterrows():
        idx = int(row["fits_index"])
        rows.append(
            [
                idx,
                row["scan"],
                f"{row['sam_th']:.4f}",
                f"{row['energy']:g}",
                f"{row['exposure']:g}",
                int(row.get("n_sat_roi", 0)),
                int(row.get("n_sat_dark", 0)),
                "cost a stitch point" if idx in costly
                else "not a stitch overlap candidate",
                f"`{loader.path_for(idx)}`",
                f"[ROI]({image_names[idx]})" if idx in image_names else "",
            ]
        )
    lines += _markdown_table(
        [
            "fits_index", "scan", "sam_th", "energy [eV]", "exposure [s]",
            "sat px (beam ROI)", "sat px (dark ROI)", "stitch impact", "file", "image",
        ],
        rows,
    )
    return lines


def _saturation_dropped_indices(points: pd.DataFrame) -> set[int]:
    """Return ``fits_index`` values of overlap candidates dropped for saturation."""
    if not len(points):
        return set()
    mask = (~points["used"]) & points["reason"].str.startswith("saturated")
    return {int(i) for i in points.loc[mask, "fits_index"]}


def dropped_points_lines(loader: PXRLoader, points: pd.DataFrame) -> list[str]:
    """Build the standalone Markdown table of every dropped overlap candidate.

    The companion to the summary's per-boundary tables: one row per rejected frame
    across the whole sample, carrying the full source path and the numbers behind
    each rejection.

    Args:
        loader: The processed loader (for file paths).
        points: :func:`~pxr_reduce.reduction.overlap_report` output.

    Returns:
        The report lines.
    """
    lines = [
        f"# Dropped stitch-overlap points - {loader.name}",
        "",
        "Every frame that was a stitch-overlap candidate but did not contribute to a "
        "fitted",
        "scale factor, and why. See `stitch_summary.md` for the per-boundary context.",
        "",
    ]
    dropped = points[~points["used"]] if len(points) else points
    rows: list[list[Any]] = []
    for _, row in dropped.iterrows():
        idx = int(row["fits_index"])
        rows.append(
            [
                row["scan"],
                int(row["boundary_index"]),
                f"{row['boundary_sam_th']:.4f}",
                row["side"],
                f"{row['sam_th']:.4f}",
                idx,
                f"{row['R']:.6g}",
                f"{row['counts_ratio']:.4g}",
                int(row["n_sat_roi"]),
                int(row["n_sat_dark"]),
                row["reason"],
                f"`{loader.path_for(idx)}`",
            ]
        )
    lines += _markdown_table(
        [
            "scan", "boundary", "boundary sam_th", "side", "sam_th", "fits_index",
            "R", "counts_ratio", "sat px (beam ROI)", "sat px (dark ROI)", "reason",
            "file",
        ],
        rows,
    )
    return lines


def _images_to_write(
    annotated: pd.DataFrame, points: pd.DataFrame, max_images: int
) -> list[int]:
    """Choose which saturated frames get an ROI image.

    Only frames that actually cost a stitch overlap point are rendered; every other
    saturated frame is reported in the summary text instead. If more than
    ``max_images`` qualify, the worst offenders (most saturated pixels) win and the
    shortfall is logged rather than silently dropped.

    Args:
        annotated: Per-frame annotated table.
        points: Overlap report.
        max_images: Cap on rendered frames.

    Returns:
        ``fits_index`` values to render, worst first.
    """
    costly = _saturation_dropped_indices(points)
    if not costly:
        return []
    frames = annotated[annotated["fits_index"].isin(costly)]
    if "n_sat_roi" in frames.columns:
        frames = frames.sort_values("n_sat_roi", ascending=False)
    chosen = [int(i) for i in frames["fits_index"]][:max_images]
    if len(costly) > len(chosen):
        logger.warning(
            "%d saturated frame(s) cost a stitch point but only %d ROI images were "
            "written (max_images=%d); all of them are still listed in "
            "stitch_summary.md.",
            len(costly),
            len(chosen),
            max_images,
        )
    return chosen


def save_stitch_diagnostics(
    loader: PXRLoader,
    directory: Path | str,
    *,
    dry_run: bool = False,
    max_images: int = _MAX_SATURATED_IMAGES,
) -> list[Path]:
    """Write the per-boundary stitch diagnostics for a processed loader.

    Produces one subfolder per scan holding a figure per stitch boundary and ROI
    images for saturated frames that cost an overlap point, plus a summary report and
    a CSV of every dropped candidate.

    Args:
        loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
        directory: The ``stitch/`` folder to write into (created if needed).
        dry_run: If True, log the targets and write nothing.
        max_images: Cap on saturated-ROI images (each re-reads a frame from disk).

    Returns:
        The paths written (or that would be), summary file first.

    Raises:
        RuntimeError: If the loader has not been processed.
    """
    if not loader.data_processed:
        raise RuntimeError("Call process() before save_stitch_diagnostics().")

    directory = Path(directory)
    # Annotate once and share it: each report would otherwise re-run the whole
    # normalize -> mark -> scale chain (and re-emit its warnings).
    annotated = reduction.annotate(loader.data, loader.config)
    boundaries = reduction.diagnose_stitches(
        loader.data, loader.config, annotated=annotated
    )
    points = reduction.overlap_report(
        loader.data, loader.config, annotated=annotated
    )

    # Resolve every output name before writing, so the summary can reference them.
    plot_names: dict[Any, str] = {}
    for scan_id, scan_boundaries in _by_scan(boundaries):
        scan_name = _scan_dir_name(scan_id)
        for ordinal, pos in enumerate(scan_boundaries.index, start=1):
            plot_names[pos] = f"{scan_name}/{_boundary_plot_name(ordinal)}"

    by_index = annotated.set_index("fits_index", drop=False)
    image_names: dict[int, str] = {}
    for idx in _images_to_write(annotated, points, max_images):
        scan_name = _scan_dir_name(by_index.loc[idx, "scan"])
        image_names[idx] = f"{scan_name}/saturated/frame_{idx:05d}_roi.png"

    written: list[Path] = [directory / "stitch_summary.md"]
    if len(points):
        written.append(directory / "dropped_points.md")
    written += [directory / name for name in plot_names.values()]
    written += [directory / name for name in image_names.values()]

    if dry_run:
        for target in written:
            logger.info("[dry-run] Would write %s", target)
        return written

    # --- Per-boundary figures --------------------------------------------------
    for scan_id, scan_boundaries in _by_scan(boundaries):
        group = annotated[annotated["scan"] == scan_id].reset_index(drop=True)
        for pos, boundary in scan_boundaries.iterrows():
            target = directory / plot_names[pos]
            target.parent.mkdir(parents=True, exist_ok=True)
            mine = points[
                (points["scan"] == scan_id)
                & (points["boundary_index"] == boundary["boundary_index"])
            ]
            fig = boundary_figure(group, mine, boundary, sample=loader.name)
            fig.savefig(target, dpi=150)
            logger.info("Wrote %s", target)

    # --- Saturated-frame ROI images -------------------------------------------
    for idx, name in image_names.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig = saturated_roi_figure(loader, idx, by_index.loc[idx])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not render ROI image for frame %d: %s", idx, exc)
            continue
        fig.savefig(target, dpi=150)
        logger.info("Wrote %s", target)

    # --- Markdown reports -----------------------------------------------------
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "stitch_summary.md"
    summary.write_text(
        "\n".join(
            summary_lines(
                loader, boundaries, points, annotated, plot_names, image_names
            )
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary)

    if len(points):
        target = directory / "dropped_points.md"
        target.write_text(
            "\n".join(dropped_points_lines(loader, points)) + "\n", encoding="utf-8"
        )
        logger.info("Wrote %s", target)

    return written
