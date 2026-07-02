"""Command-line interface for quick, notebook-free reductions.

``pxr-reduce run FOLDER`` loads the FITS files in a folder, reduces them with the
given (or default) parameters, and writes a ``.dat`` file plus I-vs-q plots so
data can be diagnosed as it comes in. A ``--quick`` mode skips stitch scaling and
can subsample frames for a fast first look.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

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

    config = ReductionConfig(
        detector=detector,
        energy_offset=energy_offset,
        sam_th_offset=sam_th_offset,
        dezinger=not no_dezinger,
        roi_from_beam_fit=fit_roi,
        roi_n_sigma=roi_n_sigma,
    )
    # Only override ROI defaults when the flags are explicitly provided.
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


if __name__ == "__main__":
    app()
