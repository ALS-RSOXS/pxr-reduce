"""Command-line interface for quick, notebook-free reductions.

``pxr-reduce run FOLDER`` loads the FITS files in a folder, reduces them with the
given (or default) parameters, and writes a ``.dat`` file plus I-vs-q plots so
data can be diagnosed as it comes in. A ``--quick`` mode skips stitch scaling and
can subsample frames for a fast first look.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import pandas as pd

# NOTE: heavy scientific imports (pandas, scipy, astropy, matplotlib) are done
# lazily inside the commands so the CLI prints feedback immediately instead of
# pausing on import before any output appears.

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Reduce polarized X-ray reflectivity FITS data.",
    no_args_is_help=True,
    add_completion=False,
)


def _configure_logging(verbose: bool) -> None:
    """Configure logging, scoping verbosity to pxr-reduce's own loggers.

    The root stays at WARNING so noisy third-party libraries (matplotlib's font
    manager and ticker, PIL) do not flood the output at ``-v``; only pxr-reduce
    loggers emit INFO/DEBUG.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("pxr_reduce").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


_MAX_STITCH_ECHO_LINES = 10


def _echo_stitch_quality(report: pd.DataFrame | None, prefix: str = "") -> None:
    """Echo the stitch-quality summary, naming every questionable boundary.

    Printed to stdout rather than logged so it is visible at the default verbosity —
    a mis-scaled stitch is the failure most likely to go unnoticed.

    Args:
        report: Per-boundary diagnostics from the reduced dataset, or None when no
            stitch scaling was applied.
        prefix: Prefix for each line (e.g. ``"[dry-run] "``).
    """
    import pandas as pd

    from pxr_reduce.reduction import summarize_stitches

    counts = summarize_stitches(report)
    if report is None or not counts["total"]:
        return
    typer.echo(
        f"{prefix}Stitches: {counts['total']} ({counts['ok']} ok, "
        f"{counts['suspect']} suspect, {counts['failed']} failed)"
    )

    failed = report[report["failed"]]
    suspect = report[report["suspect"] & ~report["failed"]]
    flagged = pd.concat([failed, suspect], ignore_index=True)
    for _, row in flagged.head(_MAX_STITCH_ECHO_LINES).iterrows():
        tag = "FAILED " if row["failed"] else "SUSPECT"
        detail = row["fail_reason"] if row["failed"] else row["quality_note"]
        typer.echo(
            f"{prefix}  {tag} scan {row['scan']} th={row['sam_th']:.4f} "
            f"E={row['energy']:g} eV: {detail}"
        )
    if len(flagged) > _MAX_STITCH_ECHO_LINES:
        typer.echo(
            f"{prefix}  ... and {len(flagged) - _MAX_STITCH_ECHO_LINES} more; see "
            "the .dat header or --diagnostics"
        )


@app.command("list-detectors")
def list_detectors() -> None:
    """List the registered detector names."""
    from pxr_reduce.detectors import available_detectors

    for detector_name in available_detectors():
        typer.echo(detector_name)


@app.command()
def run(
    folder: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Folder containing .fits files."
    ),
    pattern: str = typer.Option("*.fits", "--pattern", help="Glob for FITS files."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        help="Load reduction parameters from a TOML config file (its [reduction] "
        "section). Config-setting flags below are ignored when this is given; "
        "--roi-height/--roi-width still override it.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit output .dat path (overrides --results-dir).",
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        help="Directory for outputs, relative to the current working directory "
        "(default: ./results). Not the data folder.",
    ),
    detector: str = typer.Option("default", "--detector", help="Detector spec name."),
    roi_height: int | None = typer.Option(
        None, "--roi-height", help="Beam ROI height [px] (default: config value)."
    ),
    roi_width: int | None = typer.Option(
        None, "--roi-width", help="Beam ROI width [px] (default: config value)."
    ),
    fit_roi: bool = typer.Option(
        False,
        "--fit-roi",
        help="Size the ROI from a moments fit of the direct-beam (i0) frames.",
    ),
    roi_n_sigma: float = typer.Option(
        3.0, "--roi-n-sigma", help="ROI half-extent in beam sigmas when --fit-roi."
    ),
    energy_offset: float = typer.Option(0.0, "--energy-offset", help="Energy offset [eV]."),
    sam_th_offset: float | None = typer.Option(
        None, "--sam-th-offset", help="Sample-theta offset [deg]; auto if unset."
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Skip stitch scaling (fast preview, avoids overlap pitfalls)."
    ),
    no_dezinger: bool = typer.Option(
        False,
        "--no-dezinger",
        help="Skip median-filter/dezinger for a much faster (noisier) reduction.",
    ),
    subsample: int = typer.Option(
        1, "--subsample", min=1, help="Load every Nth frame for a faster look."
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Do not export plot PNGs."),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="Do not average duplicate (theta, energy, pol) points."
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Also write diagnostic plots (counts-vs-theta per energy/pol, beam "
        "track) to a <sample>_diagnostics/ folder.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be written without writing."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Reduce all FITS files in FOLDER and write a .dat file (and plots)."""
    _configure_logging(verbose)

    typer.echo(f"Found data folder: {folder}")
    files = sorted(folder.glob(pattern))
    if subsample > 1:
        files = files[::subsample]
        logger.info("Subsampling every %d frames -> %d files", subsample, len(files))
    if not files:
        raise typer.BadParameter(f"No files matching {pattern!r} in {folder}.")
    typer.echo(f"Matched {len(files)} file(s) with pattern {pattern!r}.")
    typer.echo("Initializing (loading scientific libraries)...")

    # Heavy imports deferred here so the messages above appear immediately.
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader
    from pxr_reduce.dataset import ReducedDataset

    if config_path is not None:
        typer.echo(f"Loading reduction config from {config_path}")
        from pxr_reduce.run_config import load_run_config

        config = load_run_config(config_path).reduction
    else:
        config = ReductionConfig(
            detector=detector,
            energy_offset=energy_offset,
            sam_th_offset=sam_th_offset,
            dezinger=not no_dezinger,
            roi_from_beam_fit=fit_roi,
            roi_n_sigma=roi_n_sigma,
        )
    # ROI flags override in both cases (handy for a quick tweak of a saved config).
    if roi_height is not None:
        config.roi_height = roi_height
    if roi_width is not None:
        config.roi_width = roi_width

    typer.echo("Reading FITS headers and building metadata table...")
    loader = PXRLoader(files, config)
    typer.echo(f"Sample identified as {loader.name!r}. Processing images...")
    loader.process()
    dataset = ReducedDataset.from_loader(
        loader, apply_scale=not quick, drop_duplicates=not no_dedup
    )

    # Default outputs go to <cwd>/results, NOT the data folder. --output wins.
    if output is not None:
        out_path = output
    else:
        results_root = results_dir if results_dir.is_absolute() else Path.cwd() / results_dir
        out_path = results_root / f"{loader.name.rstrip('_-')}.dat"
    result = dataset.save(out_path, plots=not no_plots, dry_run=dry_run)

    prefix = "[dry-run] " if dry_run else ""
    typer.echo(f"{prefix}Reduced {len(loader)} frames -> {len(dataset.data)} points")
    typer.echo(f"{prefix}Wrote: {result['dat']}")
    for plot in result["plots"]:
        typer.echo(f"{prefix}Plot : {plot}")
    _echo_stitch_quality(dataset.stitch_report, prefix)

    if diagnostics:
        from pxr_reduce import diagnostics as diag

        diag_dir = Path(out_path).with_suffix("").parent / f"{Path(out_path).stem}_diagnostics"
        for path in diag.save_diagnostics(loader, diag_dir, dry_run=dry_run):
            typer.echo(f"{prefix}Diag : {path}")


@app.command()
def batch(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="TOML config file. Defaults to ./reduction_config.toml, then "
        "built-in defaults.",
    ),
    samples: list[str] = typer.Option(
        None,
        "--sample",
        "-s",
        help="Reduce only these sample name(s); repeatable. Default: all.",
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Also write diagnostic plots (counts-vs-theta, beam track) per sample.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what each sample would load and write."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Reduce every sample defined in a TOML config into per-sample .dat files."""
    _configure_logging(verbose)

    from pxr_reduce.run_config import load_run_config, resolve_config_path

    resolved = resolve_config_path(config_path)
    config = load_run_config(resolved)
    typer.echo(f"Config: {resolved or 'built-in defaults'}")
    typer.echo(f"Parent : {config.parent_dir}")

    names = samples or list(config.samples)
    if not names:
        raise typer.BadParameter("No samples defined in [samples] (or via --sample).")

    from pxr_reduce.batch import plan_batch, run_batch

    if dry_run:
        for item in plan_batch(config, names):
            typer.echo(
                f"[dry-run] {item['sample']}: scans {item['scans']} -> "
                f"{item['n_files']} file(s) -> {item['output']}"
            )
        return

    typer.echo(f"Reducing {len(names)} sample(s) with the {config.tracker!r} tracker...")
    results = run_batch(config, names, diagnostics_plots=diagnostics)
    for name, result in results.items():
        if "error" in result:
            typer.echo(f"FAILED {name}: {result['error']}")
            continue
        stitches = result.get("stitches", {})
        note = ""
        if stitches.get("suspect") or stitches.get("failed"):
            note = (
                f"  [stitches: {stitches['suspect']} suspect, "
                f"{stitches['failed']} failed - check the .dat header]"
            )
        typer.echo(f"Wrote  {name} -> {result['dat']}{note}")


@app.command("scan-samples")
def scan_samples(
    parent: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Parent folder to search recursively."
    ),
    pattern: str = typer.Option("*.fits", "--pattern", help="Glob for FITS files."),
    scan_regex: str | None = typer.Option(
        None,
        "--scan-regex",
        help="Regex with a 'scan' group, for a convention the automatic "
        "moving-vs-static block analysis cannot handle.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Discover scans under PARENT and print a ready-to-paste [samples] map."""
    _configure_logging(verbose)

    from pxr_reduce.discovery import discover_samples, suggest_sample_map

    scans = discover_samples(parent, glob=pattern, regex=scan_regex)
    if not scans:
        typer.echo(f"No FITS scans found under {parent}.")
        return
    typer.echo(f"Found {len(scans)} scan(s):")
    for scan_id, files in scans.items():
        typer.echo(f"  {scan_id}: {len(files)} file(s)")

    typer.echo("\nSuggested [samples] (paste into your config, edit names):\n")
    typer.echo("[samples]")
    for name, ids in suggest_sample_map(parent, glob=pattern, regex=scan_regex).items():
        typer.echo(f"{name} = {ids}")


@app.command("init-config")
def init_config(
    path: Path = typer.Argument(
        Path("reduction_config.toml"),
        help="Destination for the starter config (won't overwrite).",
    ),
) -> None:
    """Write a documented starter TOML config to PATH."""
    from pxr_reduce.run_config import write_default_config

    try:
        written = write_default_config(path)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {written}. Edit [paths] and [samples], then `pxr-reduce batch`.")


if __name__ == "__main__":
    app()
