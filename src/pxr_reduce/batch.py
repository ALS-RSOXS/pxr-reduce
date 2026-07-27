"""Batch reduction: turn a :class:`RunConfig` into per-sample ``.dat`` files.

Each ``[samples]`` entry names a sample and lists the scan IDs that compose it.
For every sample, the frames from all its scans are pooled into one loader (the
configured tracker), reduced, and written to ``results_root/<sample>.dat`` (plus
plots). One failing sample is logged and skipped, not fatal to the batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pxr_reduce.core import PXRLoader
from pxr_reduce.dataset import ReducedDataset
from pxr_reduce.discovery import find_scan_files
from pxr_reduce.run_config import RunConfig

logger = logging.getLogger(__name__)


def sample_files(config: RunConfig, scan_ids: list[int]) -> list[Path]:
    """Collect, in order, every FITS frame for a sample's scans.

    Args:
        config: The run configuration (supplies ``parent_dir``/glob/regex).
        scan_ids: Scan IDs composing the sample, in the order to concatenate.

    Returns:
        The pooled FITS paths (scan by scan, frame-ordered within each scan).
    """
    files: list[Path] = []
    for scan_id in scan_ids:
        found = find_scan_files(
            config.parent_dir,
            scan_id,
            glob=config.fits_glob,
            regex=config.scan_number_regex,
        )
        if not found:
            logger.warning("No FITS files found for scan %s.", scan_id)
        files.extend(found)
    return files


def plan_batch(
    config: RunConfig, names: list[str] | None = None
) -> list[dict[str, Any]]:
    """Return, without processing, what each sample would load and write.

    Args:
        config: The run configuration.
        names: Sample names to include; None uses every sample in the config.

    Returns:
        One dict per sample with ``sample``, ``scans``, ``n_files`` and ``output``.
    """
    names = names if names is not None else list(config.samples)
    plan: list[dict[str, Any]] = []
    for name in names:
        scan_ids = config.samples.get(name, [])
        files = sample_files(config, scan_ids)
        plan.append(
            {
                "sample": name,
                "scans": scan_ids,
                "n_files": len(files),
                "output": config.results_root / f"{name}.dat",
            }
        )
    return plan


def reduce_sample(
    config: RunConfig, name: str, *, progress: bool = True, dry_run: bool = False
) -> dict[str, Any]:
    """Reduce one sample and write its ``.dat`` (and plots).

    Args:
        config: The run configuration.
        name: Sample name (must be a key in ``config.samples``).
        progress: Show the tracker's progress bar (simple tracker only).
        dry_run: If True, do everything except write files.

    Returns:
        The mapping returned by :meth:`ReducedDataset.save` (``dat``/``plots``).

    Raises:
        KeyError: If ``name`` is not in the config's samples.
        FileNotFoundError: If no FITS files are found for the sample.
    """
    scan_ids = config.samples[name]
    files = sample_files(config, scan_ids)
    if not files:
        raise FileNotFoundError(
            f"No FITS files found for sample {name!r} (scans {scan_ids})."
        )
    logger.info("Reducing %r: %d frame(s) from scan(s) %s.", name, len(files), scan_ids)

    # Files are pre-ordered (scan by scan, frame within scan); index by position
    # so several concatenated scans don't need a shared incrementing filename index.
    loader = PXRLoader(files, config.reduction, index_by_position=True, name=name)
    if config.tracker == "simple":
        loader.process(
            search_radius=config.search_radius,
            filter_size=config.filter_size,
            progress=progress,
        )
    else:
        loader.process_snr()

    dataset = ReducedDataset.from_loader(
        loader, apply_scale=config.apply_scale, drop_duplicates=config.drop_duplicates
    )
    return dataset.save(
        config.results_root / f"{name}.dat",
        plots=config.plots,
        angle_decimals=config.angle_decimals,
        dry_run=dry_run,
    )


def run_batch(
    config: RunConfig,
    names: list[str] | None = None,
    *,
    progress: bool = True,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Reduce every sample in the config (or a named subset).

    A sample that raises is logged and recorded as an error, so one bad sample
    does not abort the batch.

    Args:
        config: The run configuration.
        names: Sample names to reduce; None uses every sample in the config.
        progress: Show the tracker's progress bar (simple tracker only).
        dry_run: If True, process but write nothing.

    Returns:
        Mapping of sample name to its :meth:`ReducedDataset.save` result, or a
        ``{"error": ...}`` marker for samples that failed.
    """
    names = names if names is not None else list(config.samples)
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in config.samples:
            logger.warning("Sample %r is not in the config; skipping.", name)
            continue
        try:
            results[name] = reduce_sample(
                config, name, progress=progress, dry_run=dry_run
            )
        except Exception as exc:
            logger.error("Sample %r failed: %s", name, exc)
            results[name] = {"error": str(exc)}
    return results
