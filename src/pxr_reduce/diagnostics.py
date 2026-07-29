"""Diagnostic plots for a processed reduction.

Two extra plots, written into a ``<sample>_diagnostics/`` folder when the CLI is
run with ``--diagnostics``:

1. **Counts vs theta (pre-scaling)** — one per (energy, polarization). Raw
   ``counts_spot`` vs ``sam_th`` on a log axis, points coloured by stitch segment
   so re-measured overlap points are visible, saturated frames marked red, and
   each stitch boundary annotated with its fitted scale ratio and overlap count.
2. **Beam track** — one per sample. The beam ``(x, y)`` positions over the whole
   run, bounded to the trimmed image extent, with a connecting trace per scan and
   saturated frames highlighted.
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

from pxr_reduce import reduction, stitch_diagnostics

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


def beam_track_figure(loader: PXRLoader, *, sample: str) -> Figure:
    """Build the beam-spot track diagnostic for a whole sample.

    Args:
        loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
        sample: Sample name for the title.

    Returns:
        The rendered :class:`~matplotlib.figure.Figure`.
    """
    d = loader.data.sort_values("fits_index")
    ys = np.array([spot[0] for spot in d["beam_spot"]], dtype=float)
    xs = np.array([spot[1] for spot in d["beam_spot"]], dtype=float)
    saturated = d["is_saturated"].to_numpy(dtype=bool)
    scans = d["scan"].to_numpy()

    # Axes bounded to the trimmed image extent (where the beam can actually land).
    first = int(d["fits_index"].iloc[0])
    height, width = loader.get_clean_image(first).shape[:2]

    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # One connecting trace per scan (each scan restarts on a condition change).
    for i, scan_id in enumerate(sorted(set(scans))):
        mask = scans == scan_id
        ax.plot(xs[mask], ys[mask], "-", color=_CATEGORICAL(i % 10), lw=1,
                alpha=0.7, zorder=1, label=f"scan {scan_id}")

    ax.scatter(xs[~saturated], ys[~saturated], s=16, color="tab:blue",
               zorder=2, label="beam")
    if saturated.any():
        ax.scatter(xs[saturated], ys[saturated], s=36, marker="x", color="red",
                   zorder=3, label="saturated")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image convention: row 0 at top
    ax.set_aspect("equal")
    ax.set_xlabel("beam x [pix]")
    ax.set_ylabel("beam y [pix]")
    ax.set_title(f"{sample}   beam track (trimmed image {width}x{height})")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


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

    for (energy, pol), group in annotated.groupby(
        ["energy", "polarization"], sort=True
    ):
        target = directory / f"counts_vs_theta_E{energy:g}eV_P{pol:g}.png"
        written.append(target)
        if dry_run:
            logger.info("[dry-run] Would write %s", target)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        fig = counts_vs_theta_figure(
            group, sample=sample, energy=float(energy), pol=float(pol)
        )
        fig.savefig(target, dpi=150)
        logger.info("Wrote %s", target)

    target = directory / "beam_track.png"
    written.append(target)
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        beam_track_figure(loader, sample=sample).savefig(target, dpi=150)
        logger.info("Wrote %s", target)

    written += stitch_diagnostics.save_stitch_diagnostics(
        loader, directory / "stitch", dry_run=dry_run
    )
    return written
