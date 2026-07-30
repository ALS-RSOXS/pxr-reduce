"""Diagnostic plots for a processed reduction.

Two extra plots, written into a ``<sample>_diagnostics/`` folder when the CLI is
run with ``--diagnostics``:

Everything is written per **sweep** and named ``<kind>_id{scan_id}_sweep{n}_E{eV}_P{pol}``,
so any curve can be traced straight back to its files.

1. **RawCounts** — raw ``counts_spot`` vs ``sam_th`` on a log axis, points coloured by
   stitch segment so re-measured overlap points are visible, saturated frames marked
   red, and each stitch boundary annotated with its fitted scale ratio and overlap
   count.
2. **BeamTrack** — the beam ``(x, y)`` path for that sweep, bounded to the trimmed
   image extent, coloured by frame index so the direction of travel is visible and a
   drifting or jumping beam can be traced to the frames responsible.
3. **stitch/** — a per-scan breakdown of every stitch boundary: which overlap
   points were used, which were dropped and why, the fit and its quality checks,
   and ROI images of saturated frames that cost an overlap point. See
   :mod:`pxr_reduce.stitch_diagnostics`.

Figures are built with the headless Agg backend (no pyplot state machine), like
:mod:`pxr_reduce.dataset`'s plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pxr_reduce import metadata, reduction, stitch_diagnostics

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)

_CATEGORICAL = colormaps["tab10"]


def counts_vs_theta_figure(
    group: pd.DataFrame, *, sample: str, energy: float, pol: float
) -> Figure:
    """Build the raw-counts-vs-theta diagnostic for one (energy, polarization).

    Args:
        group: Annotated rows (from :func:`pxr_reduce.reduction.annotate`) for one
            energy/polarization, in any order.
        sample: Sample name for the title.
        energy: Photon energy of the group (eV).
        pol: Polarization of the group.

    Returns:
        The rendered :class:`~matplotlib.figure.Figure`.
    """
    g = group.sort_values("fits_index").reset_index(drop=True)
    theta = g["sam_th"].to_numpy(dtype=float)
    counts = g["counts_spot"].to_numpy(dtype=float)
    saturated = g["is_saturated"].to_numpy(dtype=bool)
    # Stitch segment index: increments at each marked boundary.
    segment = (g["mark"] == 1).cumsum().to_numpy()

    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Faint background (dark-ROI counts) for context on the high-angle tail.
    if "counts_dark" in g.columns:
        ax.plot(theta, g["counts_dark"], ".", color="0.8", ms=3,
                label="dark ROI", zorder=1)

    # Unsaturated points, coloured by stitch segment.
    for seg in range(int(segment.max()) + 1):
        mask = (segment == seg) & ~saturated
        if mask.any():
            ax.scatter(theta[mask], counts[mask], s=18, zorder=2,
                       color=_CATEGORICAL(seg % 10), label=f"segment {seg}")
    if saturated.any():
        ax.scatter(theta[saturated], counts[saturated], s=42, marker="x",
                   color="red", zorder=3, label="saturated")

    # Stitch boundaries: vertical line + fitted-ratio annotation.
    for b in [i for i in range(len(g)) if g["mark"].iloc[i] == 1]:
        theta_b = float(theta[b])
        ax.axvline(theta_b, ls="--", color="0.5", lw=1, zorder=1)
        prev_scale = float(g["scale"].iloc[b - 1]) if b > 0 else 1.0
        ratio = float(g["scale"].iloc[b]) / prev_scale if prev_scale else float("nan")
        npts = int(g["num_stitch_points"].iloc[b])
        # This boundary's own outcome — not failed_stitch_mask, which stays set for
        # every later boundary once one fails.
        failed = bool(g["stitch_failed"].iloc[b])
        suspect = bool(g["stitch_suspect"].iloc[b])
        if failed:
            label, color = "stitch\nFAILED", "red"
        elif suspect:
            label, color = f"?×{ratio:.3g}\n{npts} pts", "darkorange"
        else:
            label, color = f"×{ratio:.3g}\n{npts} pts", "0.25"
        ax.annotate(
            label, xy=(theta_b, 0.98), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=7, color=color,
        )

    ax.set_yscale("log")
    ax.set_xlabel("sam_th [deg]")
    ax.set_ylabel("counts_spot [ADU]  (raw, pre-scaling)")
    ax.set_title(f"{sample}   E={energy:g} eV   pol={pol:g}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def beam_track_figure(
    group: pd.DataFrame, *, sample: str, extent: tuple[int, int], tag: str
) -> Figure:
    """Build the beam-spot track diagnostic for **one sweep**.

    One figure per sweep rather than per sample: pooled samples run to thousands of
    frames across a dozen sweeps, and overlaying them hides which sweep a wandering
    beam belongs to.

    The trace is drawn in acquisition order and coloured by frame index, so the
    direction of travel is visible and a beam that jumps or drifts to an edge can be
    traced to the frames responsible.

    Args:
        group: One sweep's rows from the loader table (needs ``beam_spot``,
            ``fits_index``, ``is_saturated``).
        sample: Sample name for the title.
        extent: ``(height, width)`` of the trimmed frame, bounding the axes.
        tag: Sweep identifier for the title.

    Returns:
        The rendered :class:`~matplotlib.figure.Figure`.
    """
    d = group.sort_values("fits_index")
    ys = np.array([spot[0] for spot in d["beam_spot"]], dtype=float)
    xs = np.array([spot[1] for spot in d["beam_spot"]], dtype=float)
    saturated = d["is_saturated"].to_numpy(dtype=bool)
    height, width = extent

    fig = Figure(figsize=(10.5, 5.2), layout="constrained")
    FigureCanvasAgg(fig)
    full, zoom = fig.subplots(1, 2)

    order = np.arange(len(xs))
    points = None
    for ax in (full, zoom):
        ax.plot(xs, ys, "-", color="0.7", lw=0.8, zorder=1)
        points = ax.scatter(xs, ys, c=order, cmap="viridis", s=20, zorder=2)
        if saturated.any():
            ax.scatter(xs[saturated], ys[saturated], s=52, marker="x", color="red",
                       zorder=3, label="saturated")
        if len(xs):
            ax.plot(xs[0], ys[0], "o", mfc="none", mec="black", ms=12, zorder=4,
                    label="first frame")
        ax.set_aspect("equal")
        ax.set_xlabel("beam x [pix]")
    full.set_ylabel("beam y [pix]")
    if points is not None:
        fig.colorbar(points, ax=zoom, label="frame within sweep", fraction=0.046)

    # Left: the whole usable frame, so proximity to an edge is judgeable -- a beam
    # tracked to within half a ROI of one is where counts_spot silently reads low.
    full.set_xlim(0, width)
    full.set_ylim(height, 0)  # image convention: row 0 at top
    full.set_title(f"full trimmed frame ({width}x{height})", fontsize=9)

    # Right: the path itself. Over one sweep the beam often moves only tens of pixels,
    # which is a dot at frame scale.
    if len(xs):
        pad = max(6.0, 0.15 * max(np.ptp(xs), np.ptp(ys)))
        zoom.set_xlim(xs.min() - pad, xs.max() + pad)
        zoom.set_ylim(ys.max() + pad, ys.min() - pad)
    zoom.set_title("beam path (zoomed)", fontsize=9)
    zoom.legend(fontsize=7, loc="best")

    fig.suptitle(f"{sample}   {tag}   beam track", fontsize=10)
    return fig


def trimmed_extent(loader: PXRLoader, fits_index: int) -> tuple[int, int]:
    """Return the ``(height, width)`` of a trimmed frame, for bounding plot axes."""
    return loader.get_clean_image(fits_index).shape[:2]


def save_diagnostics(
    loader: PXRLoader, directory: Path | str, *, dry_run: bool = False
) -> list[Path]:
    """Write the diagnostic plots for a processed loader into ``directory``.

    Produces one counts-vs-theta PNG per (energy, polarization), one beam-track PNG
    for the sample, and a ``stitch/`` subfolder of per-boundary stitch diagnostics
    (see :func:`pxr_reduce.stitch_diagnostics.save_stitch_diagnostics`).

    Args:
        loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
        directory: Folder to write PNGs into (created if needed).
        dry_run: If True, log the targets and write nothing.

    Returns:
        The list of paths written (or that would be).

    Raises:
        RuntimeError: If the loader has not been processed.
    """
    if not loader.data_processed:
        raise RuntimeError("Call process() before save_diagnostics().")

    directory = Path(directory)
    sample = loader.name.rstrip("_-") or loader.name
    annotated = reduction.annotate(loader.data, loader.config)
    written: list[Path] = []

    for _, group in metadata.by_sweep(annotated):
        tag = metadata.sweep_tag_for(group)
        energy = float(group["energy"].iloc[0])
        pol = float(group["polarization"].iloc[0])

        counts_target = directory / f"RawCounts__{tag}.png"
        track_target = directory / f"BeamTrack_{tag}.png"
        written += [counts_target, track_target]
        if dry_run:
            logger.info("[dry-run] Would write %s", counts_target)
            logger.info("[dry-run] Would write %s", track_target)
            continue

        directory.mkdir(parents=True, exist_ok=True)
        counts_vs_theta_figure(
            group, sample=sample, energy=energy, pol=pol
        ).savefig(counts_target, dpi=150)
        logger.info("Wrote %s", counts_target)

        extent = trimmed_extent(loader, int(group["fits_index"].iloc[0]))
        beam_track_figure(
            group, sample=sample, extent=extent, tag=tag
        ).savefig(track_target, dpi=150)
        logger.info("Wrote %s", track_target)

    written += stitch_diagnostics.save_stitch_diagnostics(
        loader, directory / "stitch", dry_run=dry_run
    )
    return written
