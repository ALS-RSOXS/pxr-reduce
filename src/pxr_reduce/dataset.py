"""Reduced-dataset container with rich-header export, plotting, and combining.

:class:`ReducedDataset` bundles a reduced reflectivity table with its provenance
so it can be written to a ``.dat`` file with an expansive header, export I-vs-q
plot PNGs, and be combined with other datasets (e.g. two polarizations) into a
single file that preserves both sources' metadata.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import tomli_w
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from uncertainties.formatting import PDG_precision

from pxr_reduce.provenance import (
    ReductionProvenance,
    SourceProvenance,
    build_source_provenance,
)
from pxr_reduce.reduction import summarize_stitches

if TYPE_CHECKING:
    from pxr_reduce.config import ReductionConfig
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

# Decimal places to which sample-theta is trusted; also sets the assumed angular
# step (10^-N deg) used to propagate an uncertainty onto q for export rounding.
_ANGLE_DECIMALS = 4

# Cap on flagged stitch boundaries listed individually in the export header.
_MAX_STITCH_HEADER_LINES = 20


def _error_decimals(error: float) -> int | None:
    """Decimal places a value should keep given its 1-sigma ``error``.

    Applies the PDG significant-figure rule (1 or 2 figures depending on the
    error's leading digits) via :func:`uncertainties.formatting.PDG_precision`,
    then converts the retained significant figures into a decimal count.

    Args:
        error: The 1-sigma uncertainty of the value.

    Returns:
        The number of decimal places to round to (may be negative for large
        errors), or None if the error is not usable (non-finite or <= 0).
    """
    if not math.isfinite(error) or error <= 0:
        return None
    n_sig, error_rounded = PDG_precision(abs(error))
    if error_rounded <= 0:
        return None
    exponent = math.floor(math.log10(error_rounded))
    return int(n_sig - 1 - exponent)


def _round_to_decimals(value: float, decimals: int | None) -> str:
    """Format ``value`` rounded to ``decimals`` places as a plain string.

    Args:
        value: The value to format.
        decimals: Decimal places to keep; None keeps a general 6-figure
            precision (used when no usable uncertainty is available).

    Returns:
        The formatted value, free of floating-point display noise.
    """
    if not math.isfinite(value):
        return str(value)
    if decimals is None:
        return f"{value:.6g}"
    if decimals <= 0:
        return f"{round(value, decimals):.0f}"
    return f"{value:.{decimals}f}"


def _round_value_and_error(value: float, error: float) -> tuple[str, str]:
    """Round a value and its uncertainty to a shared, PDG-justified precision."""
    decimals = _error_decimals(error)
    return _round_to_decimals(value, decimals), _round_to_decimals(error, decimals)


def _q_uncertainty(q: float, sam_th_deg: float, angle_decimals: int) -> float:
    """Propagate the angular step onto q via ``q = 4pi sin(theta) / lambda``.

    Using ``dq/dtheta = q * cot(theta)`` avoids needing the wavelength: the
    angular step (one unit in the last trusted decimal, ``10**-angle_decimals``
    deg) is mapped onto a q uncertainty that sets q's export precision.

    Args:
        q: Momentum transfer in inverse angstroms.
        sam_th_deg: Sample theta in degrees.
        angle_decimals: Decimal places to which the angle is trusted.

    Returns:
        The 1-sigma q uncertainty (inverse angstroms), or NaN if theta is 0 (q
        resolution is undefined there) or q is non-finite.
    """
    theta = math.radians(sam_th_deg)
    sin_theta = math.sin(theta)
    if sin_theta == 0 or not math.isfinite(q):
        return float("nan")
    step_rad = math.radians(10.0**-angle_decimals)
    return abs(q * math.cos(theta) / sin_theta * step_rad)


def _reduction_config_toml(config: ReductionConfig) -> str:
    """Serialize a `ReductionConfig` as a ``[reduction]`` TOML table.

    Used as the embedded configuration when no full run config (``RunConfig``)
    TOML is supplied (e.g. the single-folder ``run`` command or the Python API).
    """
    reduction = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in config.to_dict().items()
        if value is not None
    }
    return tomli_w.dumps({"reduction": reduction})


@dataclass
class ReducedDataset:
    """A reduced reflectivity dataset plus its provenance.

    Args:
        data: Reduced table with columns scan, energy, polarization, sam_th, q,
            R, R_err.
        provenance: Reduction- and source-level provenance for the header.
        config_toml: The full run configuration as TOML, embedded verbatim in the
            export header for reproducibility. Defaults to a ``[reduction]`` table
            built from the loader's config when not supplied.
        stitch_report: Per-boundary stitch diagnostics (from
            :func:`~pxr_reduce.reduction.diagnose_stitches`), summarized in the
            export header. None means no stitch scaling was applied.
    """

    data: pd.DataFrame
    provenance: ReductionProvenance
    config_toml: str | None = None
    stitch_report: pd.DataFrame | None = None

    @classmethod
    def from_loader(
        cls,
        loader: PXRLoader,
        *,
        reduced: pd.DataFrame | None = None,
        apply_scale: bool = True,
        drop_duplicates: bool = True,
        reduction_time: datetime | None = None,
        config_toml: str | None = None,
    ) -> ReducedDataset:
        """Build a dataset from a processed loader.

        Args:
            loader: A processed :class:`~pxr_reduce.core.PXRLoader`.
            reduced: Precomputed reduced table; if None, ``loader.reduce`` is
                called with the given options.
            apply_scale: Passed to ``loader.reduce`` when ``reduced`` is None.
            drop_duplicates: Passed to ``loader.reduce`` when ``reduced`` is None.
            reduction_time: Timestamp to record; defaults to now.
            config_toml: Full run-config TOML to embed in the header; defaults to a
                ``[reduction]`` table built from the loader's config.

        Returns:
            A :class:`ReducedDataset`.
        """
        if reduced is None:
            reduced = loader.reduce(
                apply_scale=apply_scale, drop_duplicates=drop_duplicates
            )
        source = build_source_provenance(loader, reduced)
        provenance = ReductionProvenance.create(
            [source], reduction_time=reduction_time, cwd=loader.path
        )
        if config_toml is None:
            config_toml = _reduction_config_toml(loader.config)
        # Only meaningful when scaling ran; a quick reduction has no stitches.
        stitch_report = None
        if apply_scale:
            stitch_report = loader.diagnose_stitches()
            if len(stitch_report):
                stitch_report["sample"] = loader.name
        return cls(
            data=reduced.copy(),
            provenance=provenance,
            config_toml=config_toml,
            stitch_report=stitch_report,
        )

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
        reports = [
            r
            for r in (self.stitch_report, other.stitch_report)
            if r is not None and len(r)
        ]
        return ReducedDataset(
            data=data,
            provenance=provenance,
            config_toml=self.config_toml or other.config_toml,
            stitch_report=(
                pd.concat(reports, ignore_index=True) if reports else None
            ),
        )

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
        lines += _stitch_header_lines(self.stitch_report)
        if self.config_toml:
            lines.append("")
            lines.append("Configuration (TOML)")
            lines.append("-" * 40)
            lines.extend(self.config_toml.rstrip("\n").splitlines())
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

    def _formatted_body(self, angle_decimals: int = _ANGLE_DECIMALS) -> pd.DataFrame:
        """Return the export table with q/R/R_err/sam_th rounded for output.

        Rounding is applied only to the exported copy; :attr:`data` keeps full
        precision (so plots are unaffected). ``R`` is rounded to the precision
        justified by ``R_err`` (PDG rule); ``q`` is rounded to the precision
        justified by the angular step propagated onto it; ``sam_th`` is fixed to
        ``angle_decimals`` places.

        Args:
            angle_decimals: Decimal places to which sample theta is trusted.

        Returns:
            A copy of the ordered data with those columns as formatted strings.
        """
        df = self._ordered_data().copy()
        if "R" in df.columns and "R_err" in df.columns:
            rounded = [
                _round_value_and_error(r, e)
                for r, e in zip(df["R"], df["R_err"], strict=True)
            ]
            df["R"] = [r for r, _ in rounded]
            df["R_err"] = [e for _, e in rounded]
        # q uses the still-numeric sam_th, so round q before formatting sam_th.
        if "q" in df.columns and "sam_th" in df.columns:
            df["q"] = [
                _round_to_decimals(
                    q, _error_decimals(_q_uncertainty(q, th, angle_decimals))
                )
                for q, th in zip(df["q"], df["sam_th"], strict=True)
            ]
        if "sam_th" in df.columns:
            df["sam_th"] = [f"{v:.{angle_decimals}f}" for v in df["sam_th"]]
        return df

    def save_dat(
        self,
        path: Path | str,
        *,
        angle_decimals: int = _ANGLE_DECIMALS,
        dry_run: bool = False,
    ) -> Path:
        """Write the dataset to a tab-delimited ``.dat`` file with header.

        Values are rounded to their significant figures on export: ``R`` to the
        precision justified by ``R_err``, ``q`` to the precision justified by the
        angular step, and ``sam_th`` to ``angle_decimals`` places.

        Args:
            path: Output path (``.dat`` suffix added if missing).
            angle_decimals: Decimal places to which sample theta is trusted; also
                sets the angular step propagated onto q for its rounding.
            dry_run: If True, log the target and write nothing.

        Returns:
            The path that was (or would be) written.
        """
        path = Path(path).with_suffix(".dat")
        header = "\n".join(f"# {line}" for line in self.header_lines())
        # lineterminator + newline are set explicitly: pandas emits "\r\n" by
        # default and write_text would translate the "\n" again on Windows,
        # yielding "\r\r\n" (a stray CR / blank line) per row. Force clean "\n".
        body = self._formatted_body(angle_decimals).to_csv(
            sep="\t", index=False, lineterminator="\n"
        )
        content = f"{header}\n{body}"
        if dry_run:
            logger.info("[dry-run] Would write %d points to %s", len(self.data), path)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
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
        self,
        path: Path | str,
        *,
        plots: bool = True,
        angle_decimals: int = _ANGLE_DECIMALS,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Write the ``.dat`` file and (optionally) a sibling plots folder.

        The plots folder is named ``<stem>_plots`` next to the ``.dat`` file and
        is referenced by the returned mapping.

        Args:
            path: Output ``.dat`` path.
            plots: If True, also export I-vs-q PNGs.
            angle_decimals: Decimal places to which sample theta is trusted (see
                :meth:`save_dat`).
            dry_run: If True, log targets and write nothing.

        Returns:
            Mapping with keys ``dat`` (Path) and ``plots`` (list[Path]).
        """
        path = Path(path).with_suffix(".dat")
        dat_path = self.save_dat(path, angle_decimals=angle_decimals, dry_run=dry_run)
        plot_paths: list[Path] = []
        if plots:
            plot_dir = path.parent / f"{path.stem}_plots"
            plot_paths = self.save_plots(plot_dir, dry_run=dry_run)
        return {"dat": dat_path, "plots": plot_paths}


def _stitch_header_lines(report: pd.DataFrame | None) -> list[str]:
    """Build the header block summarizing stitch quality.

    Only boundaries that failed or came out suspect are listed, so a clean
    reduction collapses to one count line while a questionable one names every
    problem — the header should never imply a stitch was fine when it was not.

    Args:
        report: Per-boundary diagnostics, or None if no scaling was applied.

    Returns:
        Commented-header lines (without the leading ``#``).
    """
    lines = ["", "Stitch quality", "-" * 40]
    if report is None:
        return lines + ["Stitch scaling : not applied (quick reduction)"]
    if not len(report):
        return lines + ["Boundaries     : 0 (no stitch boundaries detected)"]

    counts = summarize_stitches(report)
    lines.append(
        f"Boundaries     : {counts['total']} ({counts['ok']} ok, "
        f"{counts['suspect']} suspect, {counts['failed']} failed)"
    )
    failed = report[report["failed"]]
    suspect = report[report["suspect"] & ~report["failed"]]

    flagged = pd.concat([failed, suspect], ignore_index=True)
    multi_sample = "sample" in report.columns and report["sample"].nunique() > 1
    for _, row in flagged.head(_MAX_STITCH_HEADER_LINES).iterrows():
        tag = "FAILED " if row["failed"] else "SUSPECT"
        detail = row["fail_reason"] if row["failed"] else row["quality_note"]
        where = f"{row['sample']} " if multi_sample else ""
        lines.append(
            f"  {tag} {where}scan {row['scan']} th={row['sam_th']:.4f} "
            f"E={row['energy']:g} eV pol={row['polarization']:g}: {detail}"
        )
    if len(flagged) > _MAX_STITCH_HEADER_LINES:
        lines.append(
            f"  ... and {len(flagged) - _MAX_STITCH_HEADER_LINES} more flagged "
            "boundaries; run with --diagnostics for the full picture"
        )
    return lines


def _source_header_lines(index: int, src: SourceProvenance) -> list[str]:
    """Build the header lines describing one source.

    The reduction parameters are recorded once in the embedded ``Configuration
    (TOML)`` block, so this per-source summary omits them.
    """
    energies = ", ".join(f"{e:g}" for e in src.energies)
    pols = ", ".join(f"{p:g}" for p in src.polarizations)
    detector_name = src.config.get("detector_name", "?")
    noise_measured = src.config.get("detector_noise_measured", False)
    noise_note = "" if noise_measured else "  [PLACEHOLDER noise specs]"
    lines = [
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
    ]
    if src.header_override:
        # Records that the per-frame metadata did not come from the FITS files.
        lines += [
            "Metadata source: header files, NOT the FITS headers",
            f"  {src.header_override}",
            "  nominal (Goal) values drive corrections; q uses the readback (Actual)",
        ]
    return lines


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
