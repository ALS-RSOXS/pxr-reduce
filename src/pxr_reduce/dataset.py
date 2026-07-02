"""Reduced-dataset container with rich-header export, plotting, and combining.

:class:`ReducedDataset` bundles a reduced reflectivity table with its provenance
so it can be written to a ``.dat`` file with an expansive header, export I-vs-q
plot PNGs, and be combined with other datasets (e.g. two polarizations) into a
single file that preserves both sources' metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pxr_reduce.provenance import (
    ReductionProvenance,
    SourceProvenance,
    build_source_provenance,
)

if TYPE_CHECKING:
    from pxr_reduce.core import PXRLoader

logger = logging.getLogger(__name__)

# Output column order and units for the .dat file.
_DAT_COLUMNS = ["q", "R", "R_err", "energy", "polarization", "sam_th", "scan"]
_COLUMN_UNITS = {
    "q": "A^-1",
    "R": "arb",
    "R_err": "arb",
    "energy": "eV",
    "polarization": "deg_or_pct",
    "sam_th": "deg",
    "scan": "-",
}


@dataclass
class ReducedDataset:
    """A reduced reflectivity dataset plus its provenance.

    Args:
        data: Reduced table with columns scan, energy, polarization, sam_th, q,
            R, R_err.
        provenance: Reduction- and source-level provenance for the header.
    """

    data: pd.DataFrame
    provenance: ReductionProvenance

    @classmethod
    def from_loader(
        cls,
        loader: PXRLoader,
        *,
        reduced: pd.DataFrame | None = None,
        apply_scale: bool = True,
        drop_duplicates: bool = True,
        reduction_time: datetime | None = None,
    ) -> ReducedDataset:
        """Build a dataset from a processed loader.

        Args:
            loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
            reduced: Precomputed reduced table; if None, ``loader.reduce`` is
                called with the given options.
            apply_scale: Passed to ``loader.reduce`` when ``reduced`` is None.
            drop_duplicates: Passed to ``loader.reduce`` when ``reduced`` is None.
            reduction_time: Timestamp to record; defaults to now.

        Returns:
            A :class:`ReducedDataset`.
        """
        if reduced is None:
            reduced = loader.reduce(
                apply_scale=apply_scale, drop_duplicates=drop_duplicates
            )
        source = build_source_provenance(loader)
        provenance = ReductionProvenance.create(
            [source], reduction_time=reduction_time, cwd=loader.path
        )
        return cls(data=reduced.copy(), provenance=provenance)

    def combine(self, other: ReducedDataset) -> ReducedDataset:
        """Combine this dataset with another, preserving both provenances.

        Rows are concatenated; both sources' provenance is retained so the header
        documents every input (e.g. two polarizations).

        Args:
            other: The dataset to merge in.

        Returns:
            A new combined :class:`ReducedDataset`.
        """
        merged_sources = self.provenance.sources + other.provenance.sources
        provenance = ReductionProvenance(
            reduction_time=datetime.now().isoformat(),
            software_version=self.provenance.software_version,
            git_commit=self.provenance.git_commit,
            uncertainty_model=self.provenance.uncertainty_model,
            sources=merged_sources,
        )
        data = pd.concat([self.data, other.data], ignore_index=True)
        return ReducedDataset(data=data, provenance=provenance)

    @staticmethod
    def combine_all(datasets: list[ReducedDataset]) -> ReducedDataset:
        """Combine a list of datasets into one.

        Args:
            datasets: Datasets to merge (at least one).

        Returns:
            The combined dataset.

        Raises:
            ValueError: If ``datasets`` is empty.
        """
        if not datasets:
            raise ValueError("combine_all requires at least one dataset.")
        result = datasets[0]
        for other in datasets[1:]:
            result = result.combine(other)
        return result

    # -- Header ---------------------------------------------------------------

    def header_lines(self) -> list[str]:
        """Return the commented header lines for the .dat file."""
        p = self.provenance
        lines = [
            "pxr-reduce reduced dataset",
            "=" * 40,
            f"Reduction time : {p.reduction_time}",
            f"Software       : pxr-reduce {p.software_version}"
            + (f" (git {p.git_commit})" if p.git_commit else ""),
            f"Uncertainty    : {p.uncertainty_model}",
            f"Data points    : {len(self.data)}",
            f"Sources        : {len(p.sources)}",
        ]
        for i, src in enumerate(p.sources, start=1):
            lines += _source_header_lines(i, src)
        lines.append("")
        lines.append("Columns: " + " ".join(_DAT_COLUMNS))
        lines.append(
            "Units  : " + " ".join(_COLUMN_UNITS[c] for c in _DAT_COLUMNS)
        )
        return lines

    # -- Writing --------------------------------------------------------------

    def _ordered_data(self) -> pd.DataFrame:
        cols = [c for c in _DAT_COLUMNS if c in self.data.columns]
        return self.data[cols]

    def save_dat(self, path: Path | str, *, dry_run: bool = False) -> Path:
        """Write the dataset to a tab-delimited ``.dat`` file with header.

        Args:
            path: Output path (``.dat`` suffix added if missing).
            dry_run: If True, log the target and write nothing.

        Returns:
            The path that was (or would be) written.
        """
        path = Path(path).with_suffix(".dat")
        header = "\n".join(f"# {line}" for line in self.header_lines())
        body = self._ordered_data().to_csv(sep="\t", index=False)
        content = f"{header}\n{body}"
        if dry_run:
            logger.info("[dry-run] Would write %d points to %s", len(self.data), path)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote %d points to %s", len(self.data), path)
        return path

    def save_plots(self, directory: Path | str, *, dry_run: bool = False) -> list[Path]:
        """Write one I-vs-q PNG per (energy, polarization) group.

        Args:
            directory: Folder to write PNGs into (created if needed).
            dry_run: If True, log the targets and write nothing.

        Returns:
            The list of PNG paths written (or that would be).
        """
        directory = Path(directory)
        written: list[Path] = []
        groups = self.data.groupby(["energy", "polarization"], sort=True)
        for (energy, pol), group in groups:
            fname = f"IvsQ_E{energy:g}eV_P{pol:g}.png"
            target = directory / fname
            written.append(target)
            if dry_run:
                logger.info("[dry-run] Would write plot %s", target)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            _write_ivsq_plot(group, energy, pol, target)
            logger.info("Wrote plot %s", target)
        return written

    def save(
        self, path: Path | str, *, plots: bool = True, dry_run: bool = False
    ) -> dict[str, Any]:
        """Write the ``.dat`` file and (optionally) a sibling plots folder.

        The plots folder is named ``<stem>_plots`` next to the ``.dat`` file and
        is referenced by the returned mapping.

        Args:
            path: Output ``.dat`` path.
            plots: If True, also export I-vs-q PNGs.
            dry_run: If True, log targets and write nothing.

        Returns:
            Mapping with keys ``dat`` (Path) and ``plots`` (list[Path]).
        """
        path = Path(path).with_suffix(".dat")
        dat_path = self.save_dat(path, dry_run=dry_run)
        plot_paths: list[Path] = []
        if plots:
            plot_dir = path.parent / f"{path.stem}_plots"
            plot_paths = self.save_plots(plot_dir, dry_run=dry_run)
        return {"dat": dat_path, "plots": plot_paths}


def _source_header_lines(index: int, src: SourceProvenance) -> list[str]:
    """Build the header lines describing one source."""
    energies = ", ".join(f"{e:g}" for e in src.energies)
    pols = ", ".join(f"{p:g}" for p in src.polarizations)
    detector_name = src.config.get("detector_name", "?")
    noise_measured = src.config.get("detector_noise_measured", False)
    noise_note = "" if noise_measured else "  [PLACEHOLDER noise specs]"
    config_items = ", ".join(
        f"{k}={v}"
        for k, v in src.config.items()
        if not k.startswith("detector_")
    )
    return [
        "",
        f"--- Source {index} ---",
        f"Sample name    : {src.sample_name}",
        f"Source path    : {src.source_path}",
        f"Frames / scans : {src.n_frames} / {src.n_scans}",
        f"Collected      : {src.collection_time_start} .. {src.collection_time_end}",
        f"Energies [eV]  : {energies}",
        f"Polarizations  : {pols}",
        f"sam_th offset  : {src.sam_th_offset} deg",
        f"Detector       : {detector_name}{noise_note}",
        f"Config         : {config_items}",
    ]


def _write_ivsq_plot(
    group: pd.DataFrame, energy: float, pol: float, target: Path
) -> None:
    """Render and save a single I-vs-q plot using the headless Agg backend."""
    ordered = group.sort_values("q")
    fig = Figure(figsize=(6, 4.5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.errorbar(
        ordered["q"],
        ordered["R"],
        yerr=ordered["R_err"],
        fmt="o-",
        ms=3,
        lw=1,
        capsize=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"q [$\mathrm{\AA}^{-1}$]")
    ax.set_ylabel("Reflectivity [arb]")
    ax.set_title(f"E = {energy:g} eV, pol = {pol:g}")
    fig.tight_layout()
    canvas.print_png(target)
